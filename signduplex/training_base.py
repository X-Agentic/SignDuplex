#!/usr/bin/env python3
"""
FullDuplex Sign Turn-Taking — Exp1 (GPU3)
================================================================================
第一版四状态手语 turn-taking 训练 / 可学习性验证。

输入 (仅这三路, 其余 meta 严格禁止):
  1. RGB  prefix : video frames[0 : window_end_frame]  → MiniCPM-o 原生 SigLIP(vpm)+resampler (全程冻结)
  2. Pose prefix : pose  frames[0 : window_end_frame]  → Uni-Sign → TemporalAdapter → SignProjector
  3. system_speaking → <SYS_SPEAKING> / <SYS_SILENT>

输出: 单个状态 token ∈ {<COMPLETE>, <INCOMPLETE>, <BACKCHANNEL>, <WAIT>}

架构复用 Exp25 warm-start checkpoint（由环境变量指定），不重新设计:
  SignProjector 768->4096->ReLU->4096 + TemporalAdapter(learned_down2) + Qwen3 L20~35 Attn LoRA
  + Uni-Sign 解冻 pose_proj / fusion_gcn_modules
新增可训练量仅: 6 个 special token 的 embedding 行 + lm_head 行 (6*4096*2 = 49,152)

Loss: 普通 token-level CrossEntropy, 全词表 (151748+6), 仅监督 1 个位置, 其余 labels=-100
"""
import argparse, os, sys
from pathlib import Path

_p = argparse.ArgumentParser(add_help=True)
_p.add_argument('--epochs', type=int, default=5)
_p.add_argument('--rgb_frames', type=int, default=8)
_p.add_argument('--workers', type=int, default=8)
_p.add_argument('--smoke', type=int, default=0, help='>0: 只跑这么多样本做自检')
ARGS = _p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('SIGNDUPLEX_PHYSICAL_GPU', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ── 路径 / 超参 ───────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR     = os.environ.get('SIGNDUPLEX_BASE_DATA_DIR', str(REPO_ROOT / 'data' / 'streaming_chunks_final'))
UNISIGN_DIR  = os.environ.get('SIGNDUPLEX_UNISIGN_DIR', str(REPO_ROOT / 'models' / 'Uni-Sign'))
MINICPM_DIR  = os.environ.get('SIGNDUPLEX_MINICPM_DIR', str(REPO_ROOT / 'models' / 'MiniCPM-o-4_5'))
EXP25_CKPT   = os.environ.get('SIGNDUPLEX_WARM_START', str(REPO_ROOT / 'checkpoints' / 'warm_start.pth'))
UNISIGN_CKPT = f'{UNISIGN_DIR}/hf_assets/csl_daily_pose_only_slt.pth'
OUT_DIR      = os.environ.get('SIGNDUPLEX_BASE_OUTPUT_DIR', str(REPO_ROOT / 'outputs' / 'training_base'))

STATE_TOKENS = ['<COMPLETE>', '<INCOMPLETE>', '<BACKCHANNEL>', '<WAIT>']
SYS_TOKENS   = ['<SYS_SPEAKING>', '<SYS_SILENT>']
NEW_TOKENS   = STATE_TOKENS + SYS_TOKENS

BATCH_SIZE    = 1
GRAD_ACCUM    = 8
NUM_EPOCHS    = ARGS.epochs
RGB_FRAMES    = ARGS.rgb_frames
MAX_POSE_LEN  = 300
WAIT_OVERSAMPLE = 10
SEED          = 42
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.03
MAX_GRAD_NORM = 1.0

# warm-start 模块用 Exp25 的一半 lr; 新增 token 行用较大 lr (全新随机初始化)
LR_NEW_TOKENS = 1e-3
LR_PROJ       = 5e-5
LR_ADAPTER    = 2.5e-5
LR_LORA       = 1e-5
LR_POSE_PROJ  = 1.5e-5
LR_FUSION_GCN = 5e-6

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_LAYERS  = list(range(20, 36))
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

for sub in ['', '/best', '/last', '/checkpoints', '/preds']:
    Path(OUT_DIR + sub).mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f'{OUT_DIR}/train.log', encoding='utf-8'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def _patch_broken_flash_attn():
    """本机 flash_attn 编译损坏, mock 掉以便 trust_remote_code 导入通过 (同 Exp25)。"""
    import importlib.machinery
    from types import ModuleType
    def _mock(name, is_pkg=False, attrs=()):
        m = ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
        m.__loader__ = None
        m.__package__ = name if is_pkg else name.rsplit('.', 1)[0]
        if is_pkg: m.__path__ = []
        for a in attrs: setattr(m, a, None)
        sys.modules[name] = m
        return m
    _mock('flash_attn_2_cuda')
    _mock('flash_attn', is_pkg=True, attrs=('flash_attn_func', 'flash_attn_varlen_func'))
    _mock('flash_attn.flash_attn_interface',
          attrs=('flash_attn_func', 'flash_attn_varlen_func', 'flash_attn_with_kvcache'))
    _mock('flash_attn.bert_padding', attrs=('index_first_axis', 'pad_input', 'unpad_input'))
_patch_broken_flash_attn()

import copy, json, math, pickle, random, re, time
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device('cuda:0')     # 已被 CUDA_VISIBLE_DEVICES=3 映射
DTYPE  = torch.bfloat16             # LLM / 视觉塔
TDTYPE = torch.float32              # Adapter / Projector
PDTYPE = torch.float32              # Uni-Sign


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 数据: 严格只读 rgb[0:wef] / pose[0:wef] / system_speaking / label
# ══════════════════════════════════════════════════════════════════════════════
ALLOWED_FIELDS = {'sample_id', 'video_id', 'label', 'system_speaking',
                  'rgb_path', 'pose_path', 'window_end_frame', 'video_frames'}
BANNED_FIELDS  = ['text_meta', 'gloss_meta', 'semantic_id', 'strategy',
                  'decision_ratio_hint', 'streaming_step_sec', 'elapsed_sign_time_sec',
                  'motion_score_at_decision', 'motion_meta', 'decision_frame',
                  'active_start_sec', 'active_end_sec', 'decision_time_sec']


def load_jsonl(split):
    """只保留白名单字段 —— 从源头物理上杜绝 meta 泄漏进模型输入。"""
    rows = []
    with open(f'{DATA_DIR}/{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            rows.append({k: d[k] for k in ALLOWED_FIELDS if k in d})
    return rows


# ── Pose 关键点预处理 (与 Uni-Sign/datasets.py 严格一致) ──────────────────────
def _crop_scale(motion, thr=0.3):
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape), 0, None
    xmin, xmax = min(valid[:, 0]), max(valid[:, 0])
    ymin, ymax = min(valid[:, 1]), max(valid[:, 1])
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


def load_part_kp(skeletons, confs):
    thr = 0.3
    out, scale = {}, None
    for part in ['body', 'left', 'right', 'face_all']:
        kps, cfs = [], []
        for sk, cf in zip(skeletons, confs):
            sk = sk[0]; cf = cf[0]
            if part == 'body':
                kp = sk[[0] + list(range(3, 11)), :];   c = cf[[0] + list(range(3, 11))]
            elif part == 'left':
                kp = sk[91:112, :] - sk[91, :];         c = cf[91:112]
            elif part == 'right':
                kp = sk[112:133, :] - sk[112, :];       c = cf[112:133]
            else:
                fidx = [i for i in list(range(23, 40))[::2]] + list(range(83, 91)) + [53]
                kp = sk[fidx, :] - sk[53, :];           c = cf[fidx]
            kps.append(kp); cfs.append(c)
        kps = np.stack(kps, 0); cfs = np.stack(cfs, 0)
        if part == 'body':
            res, scale, _ = _crop_scale(np.concatenate([kps, cfs[..., None]], -1), thr)
        else:
            res = np.concatenate([kps, cfs[..., None]], -1)
            if scale == 0:
                res = np.zeros(res.shape)
            else:
                res[..., :2] = res[..., :2] / scale
                res = np.clip(res, -1, 1)
                res[res[..., 2] <= thr] = 0
        out[part] = torch.tensor(res, dtype=torch.float32)
    return out


def uniform_idx(end, n):
    """[0, end) 上确定性均匀取 n 个下标 (end<=n 时全取)。dev/test/train 一致, 保证可复现。"""
    if end <= n:
        return list(range(end))
    return sorted(set(int(round(i * (end - 1) / (n - 1))) for i in range(n)))


def load_pose_window(pkl_path, window_end_frame, max_len=MAX_POSE_LEN):
    """严格取绝对帧 [0 : window_end_frame); 超长时确定性均匀下采样到 max_len。"""
    pose = pickle.load(open(pkl_path, 'rb'))
    n_total = len(pose['keypoints'])
    end = min(int(window_end_frame), n_total)
    if end <= 0:
        raise ValueError(f'window_end_frame={window_end_frame} 无有效帧 ({pkl_path})')
    idx = uniform_idx(end, max_len)
    kps = load_part_kp([pose['keypoints'][i] for i in idx],
                       [pose['scores'][i]    for i in idx])
    return kps, len(idx), end, n_total


# ── RGB: video frames[0 : window_end_frame) 均匀取 RGB_FRAMES 帧 ─────────────
def load_rgb_window(video_path, window_end_frame, n_frames=RGB_FRAMES):
    import decord
    vr = decord.VideoReader(video_path, num_threads=1)
    end = min(int(window_end_frame), len(vr))
    if end <= 0:
        raise ValueError(f'window_end_frame={window_end_frame} 无有效帧 ({video_path})')
    idx = uniform_idx(end, n_frames)
    frames = vr.get_batch(idx).asnumpy()
    del vr
    return [Image.fromarray(f).convert('RGB').resize((448, 448), Image.BICUBIC)
            for f in frames], len(idx), end


_WORKER_IP = None

def get_image_processor():
    """每个 DataLoader worker 内惰性构建, 避免 fork 前占用/无法 pickle。"""
    global _WORKER_IP
    if _WORKER_IP is None:
        from transformers import AutoProcessor
        _WORKER_IP = AutoProcessor.from_pretrained(
            MINICPM_DIR, trust_remote_code=True).image_processor
    return _WORKER_IP


class TurnTakingDataset(torch.utils.data.Dataset):
    def __init__(self, rows, name):
        self.rows = rows
        self.name = name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d = self.rows[i]
        wef = d['window_end_frame']
        try:
            imgs, n_rgb, rgb_end = load_rgb_window(d['rgb_path'], wef)
            ip = get_image_processor()
            # max_slice_nums=1 → 关闭 slice, 每帧固定 64 token (video 模式惯例)
            out = ip([imgs], do_pad=True, max_slice_nums=1, return_tensors='pt')
            pv  = out['pixel_values'][0]
            tgt = out['tgt_sizes'][0]
            kps, n_pose, pose_end, n_total = load_pose_window(d['pose_path'], wef)
            return {'ok': True, 'sample_id': d['sample_id'], 'label': d['label'],
                    'system_speaking': d['system_speaking'],
                    'pixel_values': pv, 'tgt_sizes': tgt, 'n_rgb': n_rgb,
                    'kps': kps, 'n_pose': n_pose,
                    'window_end_frame': wef, 'rgb_end': rgb_end, 'pose_end': pose_end}
        except Exception as ex:
            return {'ok': False, 'sample_id': d.get('sample_id', '?'),
                    'label': d.get('label'), 'err': f'{type(ex).__name__}: {ex}'}


def collate_one(batch):
    assert len(batch) == 1, 'batch_size 必须为 1'
    return batch[0]


# ══════════════════════════════════════════════════════════════════════════════
# 手语通道模块 (与 Exp25 完全同构, 以便 state_dict 直接载入)
# ══════════════════════════════════════════════════════════════════════════════
class PoseEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.modes              = m.modes
        self.proj_linear        = m.proj_linear
        self.gcn_modules        = m.gcn_modules
        self.fusion_gcn_modules = m.fusion_gcn_modules
        self.part_para          = m.part_para
        self.pose_proj          = m.pose_proj

    def set_trainable(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.pose_proj.parameters():
            p.requires_grad = True
        for p in self.fusion_gcn_modules.parameters():
            p.requires_grad = True
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def forward(self, src):
        with torch.no_grad():
            gcn_out, body = {}, None
            for part in self.modes:
                f = self.proj_linear[part](src[part]).permute(0, 3, 1, 2)
                f = self.gcn_modules[part](f)
                if   part == 'body':     body = f
                elif part == 'left':     f = f + body[..., -2][..., None].detach()
                elif part == 'right':    f = f + body[..., -1][..., None].detach()
                elif part == 'face_all': f = f + body[...,  0][..., None].detach()
                gcn_out[part] = f
        feats = []
        for part in self.modes:
            f = self.fusion_gcn_modules[part](gcn_out[part])
            feats.append(f.mean(-1).transpose(1, 2))
        x = torch.cat(feats, dim=-1) + self.part_para
        return self.pose_proj(x)


def build_pose_encoder():
    _cwd = os.getcwd()
    os.chdir(UNISIGN_DIR)
    sys.path.insert(0, UNISIGN_DIR)
    from models import Uni_Sign
    args = Namespace(hidden_dim=256, dataset='CSL_Daily',
                     rgb_support=False, max_length=MAX_POSE_LEN, label_smoothing=0.2)
    log.info('[Uni-Sign] 构建模型...')
    full = Uni_Sign(args)
    ck = torch.load(UNISIGN_CKPT, map_location='cpu')
    missing, _ = full.load_state_dict(ck['model'], strict=False)
    pose_missing = [k for k in missing if 'mt5' not in k]
    if pose_missing:
        raise RuntimeError(f'Pose encoder 权重缺失: {pose_missing[:8]}')
    os.chdir(_cwd)
    enc = PoseEncoder(full).to(DEVICE, dtype=PDTYPE)
    enc.eval()
    tr = enc.set_trainable()
    log.info(f'  Pose Encoder 可训练 {sum(p.numel() for _, p in tr):,}')
    del full
    return enc, tr


class SignProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=4096):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act     = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x.float())))

    def arch_str(self):
        return '768->4096->RELU->4096'


class LearnedCausalDownsample(nn.Module):
    """可学习 causal stride=2 时间下采样 (Exp25)。[B,T,C] -> [B,ceil(T/2),C]"""
    def __init__(self, d_model=768):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2, padding=0)
        self.norm = nn.LayerNorm(d_model)
        self._causal_pad = 1

    def forward(self, x):
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self._causal_pad, 0))
        xt = self.conv(xt)
        return self.norm(xt.transpose(1, 2))


class TemporalAdapter(nn.Module):
    """Exp25 变体: 2层 causal TransformerEncoder (Pre-LN) + LearnedCausalDownsample(stride=2)"""
    def __init__(self, d_model=768, n_layers=2, n_heads=8, ffn=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)
        self.down = LearnedCausalDownsample(d_model)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.down(x)

    def arch_str(self):
        return ('2层 causal TransformerEncoder (Pre-LN) + '
                'LearnedCausalDownsample(stride=2): d=768, heads=8, ffn=2048, dropout=0.1')


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(p=dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features,  dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out   = self.base(x)
        h     = self.lora_dropout(x.float())
        delta = (h @ self.lora_A.t()) @ self.lora_B.t()
        return out + (delta * self.scaling).to(out.dtype)


def inject_lora(llm, layer_ids, targets, r, alpha, dropout):
    layers = llm.model.layers
    for i in layer_ids:
        attn = layers[i].self_attn
        for t in targets:
            base = getattr(attn, t)
            assert isinstance(base, nn.Linear)
            setattr(attn, t, LoRALinear(base, r, alpha, dropout).to(DEVICE))
    log.info(f'[LoRA] 注入 {len(layer_ids)*len(targets)} 个模块 '
             f'(层 {min(layer_ids)}~{max(layer_ids)}, r={r}, alpha={alpha})')


def lora_parameters(llm):
    return [(n, p) for n, p in llm.named_parameters() if '.lora_' in n]

def lora_state_dict(llm):
    return {n: p.detach().float().cpu() for n, p in lora_parameters(llm)}

def set_lora_mode(llm, training: bool):
    for m in llm.modules():
        if isinstance(m, LoRALinear):
            m.lora_dropout.train(training)


class NewTokenHead(nn.Module):
    """
    6 个新 special token 的 embedding 行 + lm_head 行 (唯一新增可训练结构)。
    base embed_tokens / lm_head / norm 全程冻结 —— 不做全参数解冻。
    """
    def __init__(self, n_new, hidden, base_embed_w, base_head_w):
        super().__init__()
        mu_e, sd_e = base_embed_w.mean().item(), base_embed_w.std().item()
        mu_h, sd_h = base_head_w.mean().item(),  base_head_w.std().item()
        self.embed = nn.Parameter(torch.randn(n_new, hidden, dtype=torch.float32) * sd_e + mu_e)
        self.head  = nn.Parameter(torch.randn(n_new, hidden, dtype=torch.float32) * sd_h + mu_h)

    def forward_logits(self, h):
        """h: [.., hidden] -> [.., n_new]"""
        return h.float() @ self.head.t()


# ══════════════════════════════════════════════════════════════════════════════
# 序列构建 / 前向
# ══════════════════════════════════════════════════════════════════════════════
_SYS_TEXT = ('你是手语实时对话的轮次决策器。根据用户的手语视频与骨架序列，'
             '以及当前系统是否正在说话，判断此刻应采取的轮次状态。')
_Q_TEXT   = '当前应采取的轮次状态是：'

TOKENS = {}      # 运行时填充: 各特殊 token 的 id
_SHAPE_LOG_N = 0


def setup_tokens(tok):
    """登记新 special token, 并解析占位符 id。base 词表不 resize, 新 id 追加在末尾。"""
    base_vocab = tok.vocab_size if hasattr(tok, 'vocab_size') else len(tok)
    base_vocab = max(base_vocab, max(tok.get_added_vocab().values()) + 1)
    TOKENS['base_vocab'] = base_vocab
    # 新 token: 直接在 base 词表之后分配连续 id (不进 tokenizer, 手工拼接 ids)
    for i, t in enumerate(NEW_TOKENS):
        TOKENS[t] = base_vocab + i
    tok.add_special_tokens({'additional_special_tokens': NEW_TOKENS})   # 便于 decode 可读
    for name, s in [('im_start', '<|im_start|>'), ('im_end', '<|im_end|>'),
                    ('img_start', '<image>'), ('img_end', '</image>')]:
        i = tok.convert_tokens_to_ids(s)
        assert i is not None and i != tok.unk_token_id and i < base_vocab, f'{s} 缺失'
        TOKENS[name] = i
    ph = tok.convert_tokens_to_ids('<unk>')
    if ph is None or ph == tok.unk_token_id or ph >= base_vocab:
        ph = tok.convert_tokens_to_ids('<|vision_pad|>')
    TOKENS['ph'] = ph
    log.info(f'[Tokens] base_vocab={base_vocab}, 占位符 id={ph}')
    for t in NEW_TOKENS:
        log.info(f'  {t:<16} -> id={TOKENS[t]} (新增)')
    return base_vocab


def build_sequence(tok, n_rgb, n_pose_tok, system_speaking, label=None):
    """
    返回 (ids, labels, rgb_slots, pose_slots, pred_pos)
      rgb_slots  : RGB 特征要写入的绝对位置 list, 长度 n_rgb*64
      pose_slots : Pose 特征要写入的绝对位置 list, 长度 n_pose_tok
      pred_pos   : 用于预测状态 token 的位置 (= len(prompt)-1)
    模型输入只含: SYS token + RGB + Pose + 固定模板文本。无任何 meta。
    """
    def t(s): return tok.encode(s, add_special_tokens=False)
    ims, ime, nl = [TOKENS['im_start']], [TOKENS['im_end']], t('\n')
    ph = TOKENS['ph']

    ids = ims + t('system') + nl + t(_SYS_TEXT) + ime + nl
    ids += ims + t('user') + nl
    # (1) system_speaking
    sys_tok = TOKENS['<SYS_SPEAKING>'] if int(system_speaking) == 1 else TOKENS['<SYS_SILENT>']
    ids += t('系统状态：') + [sys_tok] + nl
    # (2) RGB prefix
    ids += t('手语视频：')
    rgb_slots = []
    for _ in range(n_rgb):
        ids += [TOKENS['img_start']]
        rgb_slots += list(range(len(ids), len(ids) + 64))
        ids += [ph] * 64 + [TOKENS['img_end']]
    ids += nl
    # (3) Pose prefix
    ids += t('手语骨架：')
    pose_slots = list(range(len(ids), len(ids) + n_pose_tok))
    ids += [ph] * n_pose_tok + nl
    # (4) 提问
    ids += t(_Q_TEXT) + ime + nl
    ids += ims + t('assistant') + nl

    P = len(ids)
    pred_pos = P - 1
    if label is None:
        return ids, None, rgb_slots, pose_slots, pred_pos
    labels = [-100] * P + [TOKENS[label]]        # 仅监督 1 个状态 token
    ids = ids + [TOKENS[label]]
    return ids, labels, rgb_slots, pose_slots, pred_pos


def embed_ids(llm, new_head, ids_t, base_vocab):
    """base 词表走冻结 embed_tokens; 新 token 走 NewTokenHead.embed (可训练)。"""
    is_new = ids_t >= base_vocab
    safe = torch.where(is_new, torch.zeros_like(ids_t), ids_t)
    emb = llm.model.embed_tokens(safe).to(DTYPE).clone()
    if is_new.any():
        pos = is_new.nonzero(as_tuple=True)
        emb[pos] = new_head.embed[ids_t[pos] - base_vocab].to(DTYPE)
    return emb


@torch.no_grad()
def rgb_embeds(minicpm, pixel_values, tgt_sizes):
    """MiniCPM-o 原生 SigLIP(vpm)+resampler, 全程冻结。-> [n_rgb*64, 4096]"""
    data = {'pixel_values': [[p.to(DEVICE, dtype=DTYPE) for p in pixel_values]],
            'tgt_sizes': [tgt_sizes.to(DEVICE)]}
    vh = minicpm.get_vision_embedding(data)[0]        # [n_rgb, 64, 4096]
    return vh.reshape(-1, vh.shape[-1]).to(DTYPE)


def pose_embeds(pose_enc, adapter, sign_proj, kps):
    src = {k: v.unsqueeze(0).to(DEVICE, dtype=PDTYPE) for k, v in kps.items()}
    feat = pose_enc(src)                  # [1,T,768]
    feat = adapter(feat)                  # [1,ceil(T/2),768]
    return sign_proj(feat).to(DTYPE)[0]   # [T',4096]


def forward_step(batch, minicpm, llm, pose_enc, adapter, sign_proj, new_head, tok,
                 base_vocab, need_loss=True):
    """
    返回 (loss, logits_state, target_idx)
      logits_state: 4 个状态 token 的 logits (受约束解码用)
    Loss = 普通 token-level CE, 全词表 (base_vocab+6), 仅 1 个监督位置 (其余 -100)。
    在唯一被监督的位置上计算 CE, 与"全序列 CE + labels=-100 掩码"数学等价。
    """
    global _SHAPE_LOG_N
    rgb = rgb_embeds(minicpm, batch['pixel_values'], batch['tgt_sizes'])
    pose = pose_embeds(pose_enc, adapter, sign_proj, batch['kps'])
    n_pose_tok = pose.shape[0]

    ids, labels, rgb_slots, pose_slots, pred_pos = build_sequence(
        tok, batch['n_rgb'], n_pose_tok, batch['system_speaking'],
        batch['label'] if need_loss else None)
    assert len(rgb_slots) == rgb.shape[0], f'{len(rgb_slots)} vs {rgb.shape[0]}'

    ids_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    emb = embed_ids(llm, new_head, ids_t, base_vocab)
    emb[0, rgb_slots]  = rgb
    emb[0, pose_slots] = pose
    attn = torch.ones(1, emb.shape[1], dtype=torch.long, device=DEVICE)

    h = llm.model(inputs_embeds=emb, attention_mask=attn).last_hidden_state
    h_last = h[0, pred_pos]                                   # 预测状态 token 的位置
    logits = torch.cat([llm.lm_head(h_last.unsqueeze(0))[0].float(),
                        new_head.forward_logits(h_last)], dim=-1)   # [base_vocab+6]

    if _SHAPE_LOG_N < 3:
        _SHAPE_LOG_N += 1
        log.info(f'  [shape] rgb={tuple(rgb.shape)} pose={tuple(pose.shape)} '
                 f'seq_len={emb.shape[1]} pred_pos={pred_pos} vocab={logits.shape[0]}')

    state_ids = [TOKENS[s] for s in STATE_TOKENS]
    logits_state = logits[state_ids]
    loss = None
    if need_loss:
        tgt = torch.tensor([TOKENS[batch['label']]], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits.unsqueeze(0), tgt)
    return loss, logits_state.detach().float(), logits.detach().float()


# ══════════════════════════════════════════════════════════════════════════════
# 指标: Accuracy / Macro P-R-F1 / 各类 P-R-F1 / 4x4 混淆矩阵 / IB 混淆 / WAIT recall
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, classes=STATE_TOKENS):
    K = len(classes)
    ci = {c: i for i, c in enumerate(classes)}
    cm = [[0] * K for _ in range(K)]
    for t, p in zip(y_true, y_pred):
        cm[ci[t]][ci[p]] += 1
    n = len(y_true)
    correct = sum(cm[i][i] for i in range(K))
    per = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(K)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {'precision': round(prec, 6), 'recall': round(rec, 6), 'f1': round(f1, 6),
                  'support': sum(cm[i]), 'tp': tp, 'fp': fp, 'fn': fn}
    macro = {k: round(sum(per[c][k] for c in classes) / K, 6)
             for k in ['precision', 'recall', 'f1']}
    out = {'n': n, 'accuracy': round(correct / n, 6) if n else 0.0,
           'macro': macro, 'per_class': per, 'confusion_matrix': cm,
           'confusion_matrix_labels': list(classes)}
    if '<INCOMPLETE>' in ci and '<BACKCHANNEL>' in ci:
        I, B = ci['<INCOMPLETE>'], ci['<BACKCHANNEL>']
        sup_I, sup_B = sum(cm[I]), sum(cm[B])
        out['incomplete_backchannel_confusion'] = {
            'INCOMPLETE_true_pred_BACKCHANNEL': cm[I][B],
            'BACKCHANNEL_true_pred_INCOMPLETE': cm[B][I],
            'INCOMPLETE_support': sup_I, 'BACKCHANNEL_support': sup_B,
            'rate_INCOMPLETE_to_BACKCHANNEL': round(cm[I][B] / sup_I, 6) if sup_I else 0.0,
            'rate_BACKCHANNEL_to_INCOMPLETE': round(cm[B][I] / sup_B, 6) if sup_B else 0.0,
            'total_IB_confusions': cm[I][B] + cm[B][I],
        }
    if '<WAIT>' in ci:
        out['wait_recall'] = per['<WAIT>']['recall']
    return out


def fmt_metrics(m, title, extra=''):
    C = m['confusion_matrix_labels']
    L = ['', '=' * 96, f'{title}{extra}', '=' * 96,
         f'样本数 n = {m["n"]}',
         f'Accuracy        = {m["accuracy"]*100:.2f}%',
         f'Macro Precision = {m["macro"]["precision"]*100:.2f}%',
         f'Macro Recall    = {m["macro"]["recall"]*100:.2f}%',
         f'Macro F1        = {m["macro"]["f1"]*100:.2f}%', '',
         '各类指标:',
         f'  {"类别":<16}{"Precision":>11}{"Recall":>10}{"F1":>10}{"Support":>9}']
    for c in C:
        p = m['per_class'][c]
        L.append(f'  {c:<16}{p["precision"]*100:>10.2f}%{p["recall"]*100:>9.2f}%'
                 f'{p["f1"]*100:>9.2f}%{p["support"]:>9}')
    L += ['', f'{len(C)}x{len(C)} Confusion Matrix (行=真实, 列=预测):']
    head = '  ' + ' ' * 16 + ''.join(f'{c[:13]:>15}' for c in C)
    L.append(head)
    for i, c in enumerate(C):
        L.append(f'  {c:<16}' + ''.join(f'{v:>15}' for v in m['confusion_matrix'][i]))
    ib = m.get('incomplete_backchannel_confusion')
    if ib:
        L += ['', 'INCOMPLETE / BACKCHANNEL 混淆专项:',
              f'  真实 INCOMPLETE  -> 预测 BACKCHANNEL : {ib["INCOMPLETE_true_pred_BACKCHANNEL"]}'
              f' / {ib["INCOMPLETE_support"]}  ({ib["rate_INCOMPLETE_to_BACKCHANNEL"]*100:.2f}%)',
              f'  真实 BACKCHANNEL -> 预测 INCOMPLETE  : {ib["BACKCHANNEL_true_pred_INCOMPLETE"]}'
              f' / {ib["BACKCHANNEL_support"]}  ({ib["rate_BACKCHANNEL_to_INCOMPLETE"]*100:.2f}%)',
              f'  双向混淆合计: {ib["total_IB_confusions"]}']
    if 'wait_recall' in m:
        L += ['', f'WAIT Recall = {m["wait_recall"]*100:.2f}% '
                  f'({m["per_class"]["<WAIT>"]["tp"]}/{m["per_class"]["<WAIT>"]["support"]})']
    L.append('=' * 96)
    return '\n'.join(L)


@torch.no_grad()
def evaluate(loader, minicpm, llm, pose_enc, adapter, sign_proj, new_head, tok,
             base_vocab, tag, log_every=400):
    set_train_mode(sign_proj, adapter, llm, new_head, False)
    recs, tot_loss, n_ok, n_skip = [], 0.0, 0, 0
    n_unconstrained_ok = 0
    state_ids = [TOKENS[s] for s in STATE_TOKENS]
    t0 = time.time()
    for i, b in enumerate(loader):
        if not b['ok']:
            n_skip += 1; log.warning(f'  {tag} skip {b["sample_id"]}: {b["err"]}'); continue
        try:
            loss, ls, full = forward_step(b, minicpm, llm, pose_enc, adapter, sign_proj,
                                          new_head, tok, base_vocab, need_loss=True)
        except Exception as ex:
            n_skip += 1; log.warning(f'  {tag} fwd skip {b["sample_id"]}: {ex}'); continue
        if torch.isfinite(loss):
            tot_loss += loss.item(); n_ok += 1
        pred = STATE_TOKENS[int(ls.argmax().item())]           # 受约束: 仅 4 类
        if int(full.argmax().item()) in state_ids:             # 诊断: 自由解码是否落在4类内
            n_unconstrained_ok += 1
        recs.append({'sample_id': b['sample_id'], 'true': b['label'], 'pred': pred,
                     'system_speaking': b['system_speaking'],
                     'logits': [round(float(x), 4) for x in ls.tolist()]})
        if log_every and (i + 1) % log_every == 0:
            log.info(f'  {tag} {i+1}/{len(loader)}  {time.time()-t0:.0f}s')
    m = compute_metrics([r['true'] for r in recs], [r['pred'] for r in recs])
    m['loss'] = round(tot_loss / n_ok, 6) if n_ok else float('inf')
    m['n_skipped'] = n_skip
    m['unconstrained_in_state_set_rate'] = round(n_unconstrained_ok / len(recs), 6) if recs else 0.0
    return m, recs


def sub_metrics_3class(recs):
    """
    补充诊断 (不改变主指标): 仅 system_speaking=0 子集上的 3 类问题。
    因为本数据 system_speaking=1 <=> <WAIT> 完全对应, WAIT 可被该标志平凡解出,
    真实可学习性体现在 COMPLETE / INCOMPLETE / BACKCHANNEL 的三类区分上。
    """
    sub = [r for r in recs if int(r['system_speaking']) == 0]
    C3 = ['<COMPLETE>', '<INCOMPLETE>', '<BACKCHANNEL>']
    sub = [r for r in sub if r['true'] in C3]
    if not sub:
        return None
    y_pred = [r['pred'] if r['pred'] in C3 else
              C3[int(np.argmax([r['logits'][STATE_TOKENS.index(c)] for c in C3]))]
              for r in sub]
    return compute_metrics([r['true'] for r in sub], y_pred, classes=C3)


# ══════════════════════════════════════════════════════════════════════════════
# 训练模式 / warm-start / 优化器 / checkpoint
# ══════════════════════════════════════════════════════════════════════════════
def set_train_mode(sign_proj, adapter, llm, new_head, training: bool):
    sign_proj.train(training)
    adapter.train(training)
    new_head.train(training)
    set_lora_mode(llm, training)
    # Uni-Sign 始终 eval (BN/dropout 行为与 Exp25 一致)


def warm_start_from_exp25(path, sign_proj, adapter, llm, pose_enc):
    log.info(f'[Warm-start] 载入 Exp25 best: {path}')
    ck = torch.load(path, map_location='cpu')
    sign_proj.load_state_dict({k: v.to(TDTYPE) for k, v in ck['sign_projector'].items()})
    adapter.load_state_dict({k: v.to(TDTYPE) for k, v in ck['temporal_adapter'].items()})
    own = dict(llm.named_parameters())
    n_lora = 0
    with torch.no_grad():
        for n, v in ck['lora'].items():
            if n not in own:
                raise KeyError(f'LoRA 参数名不存在: {n}')
            own[n].copy_(v.to(own[n].device, dtype=own[n].dtype)); n_lora += 1
    pown = dict(pose_enc.named_parameters())
    n_uni = 0
    with torch.no_grad():
        for n, v in ck['unisign_unfrozen'].items():
            if n not in pown:
                raise KeyError(f'Uni-Sign 参数名不存在: {n}')
            pown[n].copy_(v.to(pown[n].device, dtype=pown[n].dtype)); n_uni += 1
    log.info(f'  ✓ projector + temporal_adapter + LoRA({n_lora} 张量) + '
             f'Uni-Sign({n_uni} 张量)  [Exp25 epoch={ck.get("epoch")}, '
             f'dev_loss={ck.get("dev_loss")}]')
    log.info(f'  Exp25 adapter_variant={ck.get("adapter_variant")}, cfg={ck.get("adapter_cfg")}')
    assert ck.get('adapter_variant') == 'learned_down2', 'Exp25 变体不匹配'
    return ck


def audit_trainable(sign_proj, adapter, llm, pose_enc, new_head, minicpm):
    log.info('\n' + '=' * 92)
    log.info('可训练参数审计')
    log.info('=' * 92)
    errs = []
    n_proj = sum(p.numel() for p in sign_proj.parameters() if p.requires_grad)
    n_ad   = sum(p.numel() for p in adapter.parameters()   if p.requires_grad)
    n_new  = sum(p.numel() for p in new_head.parameters()  if p.requires_grad)
    n_lora = sum(p.numel() for _, p in lora_parameters(llm))
    n_pose = sum(p.numel() for n, p in pose_enc.named_parameters() if p.requires_grad)
    log.info(f'[1] SignProjector      {n_proj:>12,}  lr={LR_PROJ:.1e}')
    log.info(f'[2] TemporalAdapter    {n_ad:>12,}  lr={LR_ADAPTER:.1e}')
    log.info(f'[3] Qwen3 LoRA         {n_lora:>12,}  lr={LR_LORA:.1e}  层={LORA_LAYERS}')
    log.info(f'[4] Uni-Sign 解冻      {n_pose:>12,}  (pose_proj/fusion_gcn)')
    log.info(f'[5] 新 token embed+head{n_new:>12,}  lr={LR_NEW_TOKENS:.1e}  '
             f'({len(NEW_TOKENS)} tokens x 4096 x 2)')
    # 冻结检查
    for tag, mod in [('embed_tokens', llm.model.embed_tokens), ('lm_head', llm.lm_head),
                     ('norm', llm.model.norm), ('vpm', minicpm.vpm),
                     ('resampler', minicpm.resampler)]:
        bad = [n for n, p in mod.named_parameters() if p.requires_grad]
        if bad: errs.append(f'{tag} 未冻结: {bad[:3]}')
    if any(p.requires_grad for p in pose_enc.gcn_modules.parameters()):
        errs.append('gcn_modules 应冻结')
    if any(not p.requires_grad for p in pose_enc.pose_proj.parameters()):
        errs.append('pose_proj 应解冻')
    log.info('[6] 冻结校验: embed_tokens / lm_head / norm / vpm / resampler / '
             'gcn_modules 均冻结 ✓')
    # 因果性数值校验
    was = adapter.training; adapter.eval()
    g = torch.Generator(device='cpu').manual_seed(0)
    with torch.no_grad():
        T0, t0 = 24, 13
        x  = torch.randn(1, T0, 768, generator=g).to(DEVICE, dtype=TDTYPE)
        y1 = adapter(x)
        x2 = x.clone()
        noise = torch.randn(1, T0 - t0, 768, generator=g).to(DEVICE, dtype=TDTYPE) * 3.0
        x2[0, t0:] += noise[0]
        y2 = adapter(x2)
        k = t0 // 2
        d_b = (y1[0, :k] - y2[0, :k]).abs().max().item()
        d_a = (y1[0, k:] - y2[0, k:]).abs().max().item()
    adapter.train(was)
    log.info(f'[7] Adapter 因果性: 扰动 t>={t0} 后前{k}个输出变化={d_b:.2e} (应≈0), '
             f'后续={d_a:.2e} (应>0)')
    if d_b > 1e-4: errs.append(f'Adapter 非因果 {d_b:.2e}')
    total = n_proj + n_ad + n_lora + n_pose + n_new
    frozen = (sum(p.numel() for p in minicpm.parameters())
              + sum(p.numel() for p in pose_enc.parameters()) - n_lora - n_pose)
    log.info('-' * 92)
    log.info(f'可训练总计 {total:,} / 冻结 {frozen:,} '
             f'(可训练占比 {total/(total+frozen)*100:.4f}%)')
    log.info('-' * 92)
    if errs:
        for e in errs: log.error(f'  ✗ {e}')
        raise RuntimeError(f'审计失败 {len(errs)} 项')
    log.info('✓ 审计通过\n' + '=' * 92)
    return {'sign_projector': n_proj, 'temporal_adapter': n_ad, 'lora': n_lora,
            'unisign_unfrozen': n_pose, 'new_token_rows': n_new,
            'total_trainable': total, 'frozen': frozen,
            'trainable_ratio_pct': round(total / (total + frozen) * 100, 6)}


def build_optimizer(sign_proj, adapter, llm, pose_enc, new_head):
    groups = [
        {'params': list(sign_proj.parameters()), 'lr': LR_PROJ,
         'weight_decay': WEIGHT_DECAY, 'name': 'sign_projector'},
        {'params': list(adapter.parameters()), 'lr': LR_ADAPTER,
         'weight_decay': WEIGHT_DECAY, 'name': 'temporal_adapter'},
        {'params': [p for _, p in lora_parameters(llm)], 'lr': LR_LORA,
         'weight_decay': 0.0, 'name': 'lora'},
        {'params': list(pose_enc.pose_proj.parameters()), 'lr': LR_POSE_PROJ,
         'weight_decay': WEIGHT_DECAY, 'name': 'pose_proj'},
        {'params': list(pose_enc.fusion_gcn_modules.parameters()), 'lr': LR_FUSION_GCN,
         'weight_decay': WEIGHT_DECAY, 'name': 'fusion_gcn_modules'},
        {'params': list(new_head.parameters()), 'lr': LR_NEW_TOKENS,
         'weight_decay': 0.0, 'name': 'new_token_rows'},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    all_p = [p for g in groups for p in g['params']]
    opt_ids = {id(p) for p in all_p}
    if len(opt_ids) != len(all_p):
        raise RuntimeError('optimizer 参数组重复')
    live = {id(p) for m in [sign_proj, adapter, llm, pose_enc, new_head]
            for p in m.parameters() if p.requires_grad}
    if live != opt_ids:
        raise RuntimeError(f'requires_grad 与 optimizer 不一致: '
                           f'漏 {len(live - opt_ids)}, 多 {len(opt_ids - live)}')
    log.info(f'[优化器] AdamW, {len(groups)} 组, 共 {sum(p.numel() for p in all_p):,} 参数 ✓')
    return opt, all_p, groups


def save_ckpt(path, epoch, sign_proj, adapter, llm, pose_enc, new_head,
              train_loss, dev_metrics, cfg):
    obj = {'epoch': epoch,
           'sign_projector': {k: v.detach().float().cpu() for k, v in sign_proj.state_dict().items()},
           'temporal_adapter': {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()},
           'lora': lora_state_dict(llm),
           'unisign_unfrozen': {n: p.detach().float().cpu()
                                for n, p in pose_enc.named_parameters() if p.requires_grad},
           'new_token_head': {k: v.detach().float().cpu() for k, v in new_head.state_dict().items()},
           'new_tokens': NEW_TOKENS, 'token_ids': {t: TOKENS[t] for t in NEW_TOKENS},
           'train_loss': train_loss, 'dev_metrics': dev_metrics, 'config': cfg}
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp); os.replace(tmp, path)


def load_ckpt_weights(path, sign_proj, adapter, llm, pose_enc, new_head):
    ck = torch.load(path, map_location='cpu')
    sign_proj.load_state_dict({k: v.to(TDTYPE) for k, v in ck['sign_projector'].items()})
    adapter.load_state_dict({k: v.to(TDTYPE) for k, v in ck['temporal_adapter'].items()})
    new_head.load_state_dict({k: v.to(TDTYPE) for k, v in ck['new_token_head'].items()})
    own = dict(llm.named_parameters())
    with torch.no_grad():
        for n, v in ck['lora'].items():
            own[n].copy_(v.to(own[n].device, dtype=own[n].dtype))
    pown = dict(pose_enc.named_parameters())
    with torch.no_grad():
        for n, v in ck['unisign_unfrozen'].items():
            pown[n].copy_(v.to(pown[n].device, dtype=pown[n].dtype))
    log.info(f'  载入 best checkpoint: epoch={ck["epoch"]}, '
             f'dev Macro-F1={ck["dev_metrics"]["macro"]["f1"]:.4f}')
    return ck


def make_loader(rows, name, shuffle=False, workers=None):
    ds = TurnTakingDataset(rows, name)
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=ARGS.workers if workers is None else workers,
        collate_fn=collate_one, pin_memory=False, drop_last=False,
        prefetch_factor=4 if (ARGS.workers if workers is None else workers) > 0 else None,
        persistent_workers=False)


def oversample_wait(rows, factor=WAIT_OVERSAMPLE):
    """
    train 阶段对 <WAIT> 做 ~10x 加权过采样 —— 只在 DataLoader 层复制索引,
    绝不修改原始 JSON 文件。确定性复制 (非随机替换采样), 保证每 epoch 覆盖全部样本。
    """
    out = []
    n_wait = 0
    for r in rows:
        if r['label'] == '<WAIT>':
            out.extend([r] * factor); n_wait += 1
        else:
            out.append(r)
    log.info(f'[过采样] <WAIT> {n_wait} 条 x{factor} -> 训练集 {len(rows)} -> {len(out)} 条 '
             f'(原始 JSON 未修改)')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    set_seed(SEED)
    log.info('=' * 96)
    log.info('FullDuplex Sign Turn-Taking — Exp1 (GPU3) 四状态可学习性验证')
    log.info(f'  数据: {DATA_DIR}')
    log.info(f'  输出: {OUT_DIR}')
    log.info(f'  GPU : CUDA_VISIBLE_DEVICES=3 -> {torch.cuda.get_device_name(0)}')
    log.info(f'  RGB 帧数={RGB_FRAMES} (每帧 64 token), Pose 上限={MAX_POSE_LEN} 帧 (->/2)')
    log.info(f'  epochs={NUM_EPOCHS}, bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, bf16')
    log.info(f'  禁止入模字段: {BANNED_FIELDS}')
    log.info('=' * 96)

    # 1. 数据
    train_rows = load_jsonl('train')
    dev_rows   = load_jsonl('dev')
    test_rows  = load_jsonl('test')
    if ARGS.smoke:
        def strat(rows, k):
            """每类取 k 条, 保证 smoke 覆盖全部 4 类 (含 WAIT) 与过采样路径。"""
            out, seen = [], {}
            for r in rows:
                c = seen.get(r['label'], 0)
                if c < k:
                    out.append(r); seen[r['label']] = c + 1
            return out
        per = max(2, ARGS.smoke // 4)
        train_rows = strat(train_rows, per)
        dev_rows   = strat(dev_rows,   max(2, per // 2))
        test_rows  = strat(test_rows,  max(2, per // 2))
        log.warning(f'[SMOKE] 分层抽样 train={len(train_rows)} dev={len(dev_rows)} '
                    f'test={len(test_rows)}')
    from collections import Counter
    for nm, rr in [('train', train_rows), ('dev', dev_rows), ('test', test_rows)]:
        log.info(f'  {nm}: {len(rr)}  {dict(Counter(r["label"] for r in rr))}')
    train_os = oversample_wait(train_rows)

    # 2. 模型
    from transformers import AutoTokenizer, AutoModel
    log.info('[MiniCPM-o-4.5] 加载 tokenizer + 模型 (bf16, eager, 不初始化 audio/tts)...')
    tok = AutoTokenizer.from_pretrained(MINICPM_DIR, trust_remote_code=True)
    minicpm = AutoModel.from_pretrained(MINICPM_DIR, trust_remote_code=True,
                                        torch_dtype=DTYPE, device_map={'': DEVICE},
                                        attn_implementation='eager',
                                        init_audio=False, init_tts=False)
    minicpm.eval()
    for p in minicpm.parameters():
        p.requires_grad = False
    llm = minicpm.llm
    assert llm.config.num_hidden_layers == 36
    H = llm.config.hidden_size
    base_vocab = setup_tokens(tok)
    inject_lora(llm, LORA_LAYERS, LORA_TARGETS, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    set_lora_mode(llm, False)

    pose_enc, _ = build_pose_encoder()
    sign_proj = SignProjector(768, H).to(DEVICE, dtype=TDTYPE)
    adapter   = TemporalAdapter().to(DEVICE, dtype=TDTYPE)
    for m in (sign_proj, adapter):
        for p in m.parameters(): p.requires_grad = True
    new_head = NewTokenHead(len(NEW_TOKENS), H,
                            llm.model.embed_tokens.weight.detach(),
                            llm.lm_head.weight.detach()).to(DEVICE, dtype=TDTYPE)
    for p in new_head.parameters(): p.requires_grad = True

    # 3. warm-start (复用 Exp25, 不重新设计架构)
    exp25 = warm_start_from_exp25(EXP25_CKPT, sign_proj, adapter, llm, pose_enc)

    # 4. 审计
    audit = audit_trainable(sign_proj, adapter, llm, pose_enc, new_head, minicpm)
    with open(f'{OUT_DIR}/trainable_params.json', 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # 5. 优化器 / 调度器
    optimizer, all_trainable, groups = build_optimizer(
        sign_proj, adapter, llm, pose_enc, new_head)
    steps_per_epoch = math.ceil(len(train_os) / GRAD_ACCUM)
    total_steps  = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(f'[调度器] cosine+warmup: steps/epoch={steps_per_epoch}, '
             f'total={total_steps}, warmup={warmup_steps}')

    cfg = {
        'exp': 'FullDuplex_Sign_Train_Exp1_GPU3', 'gpu': '3', 'seed': SEED,
        'data_dir': DATA_DIR, 'out_dir': OUT_DIR,
        'reuse_checkpoint': EXP25_CKPT,
        'task': '四状态 turn-taking 单 token 分类',
        'state_tokens': STATE_TOKENS, 'sys_tokens': SYS_TOKENS,
        'model_inputs': ['rgb[0:window_end_frame]', 'pose[0:window_end_frame]',
                         'system_speaking'],
        'excluded_from_input': BANNED_FIELDS,
        'rgb_frames': RGB_FRAMES, 'rgb_tokens_per_frame': 64,
        'max_pose_len': MAX_POSE_LEN, 'pose_downsample': 2,
        'batch_size': BATCH_SIZE, 'grad_accum': GRAD_ACCUM, 'num_epochs': NUM_EPOCHS,
        'dtype': 'bfloat16', 'loss': 'token-level CrossEntropy (单个监督位置)',
        'wait_oversample': WAIT_OVERSAMPLE, 'wait_oversample_scope': 'train only',
        'lr': {g['name']: g['lr'] for g in groups},
        'warmup_ratio': WARMUP_RATIO, 'weight_decay': WEIGHT_DECAY,
        'max_grad_norm': MAX_GRAD_NORM,
        'lora': {'r': LORA_R, 'alpha': LORA_ALPHA, 'dropout': LORA_DROPOUT,
                 'layers': LORA_LAYERS, 'targets': LORA_TARGETS},
        'n_train': len(train_rows), 'n_train_oversampled': len(train_os),
        'n_dev': len(dev_rows), 'n_test': len(test_rows),
        'best_model_selection': 'dev Macro-F1',
        'param_counts': audit,
        'exp25_source': {'epoch': exp25.get('epoch'), 'dev_loss': exp25.get('dev_loss'),
                         'adapter_variant': exp25.get('adapter_variant')},
    }
    with open(f'{OUT_DIR}/config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    dev_loader = make_loader(dev_rows, 'dev')

    # 6. 训练
    history = []
    best_f1, best_epoch = -1.0, -1
    for epoch in range(1, NUM_EPOCHS + 1):
        log.info('\n' + '=' * 96)
        log.info(f'Epoch {epoch}/{NUM_EPOCHS}')
        log.info('=' * 96)
        set_train_mode(sign_proj, adapter, llm, new_head, True)
        optimizer.zero_grad(set_to_none=True)
        loader = make_loader(train_os, 'train', shuffle=True)
        run_sum, run_cnt, micro, n_skip = 0.0, 0, 0, 0
        gnorm = 0.0
        t_ep = time.time()
        for b in loader:
            if not b['ok']:
                n_skip += 1
                if n_skip <= 20: log.warning(f'  data skip {b["sample_id"]}: {b["err"]}')
                continue
            try:
                loss, _, _ = forward_step(b, minicpm, llm, pose_enc, adapter, sign_proj,
                                          new_head, tok, base_vocab, need_loss=True)
                if not torch.isfinite(loss):
                    n_skip += 1; log.warning(f'  NaN/Inf: {b["sample_id"]}'); continue
                (loss / GRAD_ACCUM).backward()
                run_sum += loss.item(); run_cnt += 1; micro += 1
                if micro % GRAD_ACCUM == 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(all_trainable, MAX_GRAD_NORM).item()
                    optimizer.step(); scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if micro % 1000 == 0:
                    el = time.time() - t_ep
                    log.info(f'  E{epoch} {micro}/{len(train_os)} loss={run_sum/run_cnt:.4f} '
                             f'gnorm={gnorm:.3f} lr_new={optimizer.param_groups[5]["lr"]:.2e} '
                             f'{el:.0f}s ({micro/el:.2f} it/s) '
                             f'mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB')
            except torch.cuda.OutOfMemoryError:
                n_skip += 1; log.warning(f'  OOM: {b["sample_id"]}')
                optimizer.zero_grad(set_to_none=True); micro = 0
                torch.cuda.empty_cache()
            except Exception as ex:
                n_skip += 1; log.warning(f'  skip {b["sample_id"]}: {ex}')
        if micro % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        train_loss = run_sum / run_cnt if run_cnt else float('inf')
        t_train = time.time() - t_ep
        log.info(f'  E{epoch} 训练完成: loss={train_loss:.4f} 有效={run_cnt} 跳过={n_skip} '
                 f'{t_train/60:.1f}min mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB')

        # dev 评估 -> Macro-F1 选 best
        t_d = time.time()
        dm, dev_recs = evaluate(dev_loader, minicpm, llm, pose_enc, adapter, sign_proj,
                                new_head, tok, base_vocab, f'E{epoch} dev')
        log.info(f'  E{epoch} dev: loss={dm["loss"]:.4f} Acc={dm["accuracy"]*100:.2f}% '
                 f'MacroF1={dm["macro"]["f1"]*100:.2f}% WAIT_recall={dm["wait_recall"]*100:.2f}% '
                 f'({time.time()-t_d:.0f}s)')
        log.info(fmt_metrics(dm, f'Epoch {epoch} — DEV'))
        s3 = sub_metrics_3class(dev_recs)
        if s3:
            log.info(f'  [补充诊断] dev 三类 (system_speaking=0 子集) MacroF1='
                     f'{s3["macro"]["f1"]*100:.2f}% Acc={s3["accuracy"]*100:.2f}%')

        history.append({'epoch': epoch, 'train_loss': round(train_loss, 6),
                        'n_train_used': run_cnt, 'n_train_skipped': n_skip,
                        'train_time_min': round(t_train / 60, 2),
                        'dev': dm, 'dev_3class_sys0': s3,
                        'peak_mem_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        save_ckpt(f'{OUT_DIR}/checkpoints/epoch_{epoch:02d}.pth', epoch, sign_proj,
                  adapter, llm, pose_enc, new_head, train_loss, dm, cfg)
        save_ckpt(f'{OUT_DIR}/last/ckpt.pth', epoch, sign_proj, adapter, llm,
                  pose_enc, new_head, train_loss, dm, cfg)
        if dm['macro']['f1'] > best_f1:
            best_f1, best_epoch = dm['macro']['f1'], epoch
            save_ckpt(f'{OUT_DIR}/best/ckpt.pth', epoch, sign_proj, adapter, llm,
                      pose_enc, new_head, train_loss, dm, cfg)
            log.info(f'  [best] dev Macro-F1={best_f1*100:.2f}% -> best/ckpt.pth 更新')
        else:
            log.info(f'  [best] 未改善 (最好 epoch {best_epoch}, MacroF1={best_f1*100:.2f}%)')
        with open(f'{OUT_DIR}/history.json', 'w', encoding='utf-8') as f:
            json.dump({'history': history, 'best_epoch': best_epoch,
                       'best_dev_macro_f1': round(best_f1, 6)}, f,
                      ensure_ascii=False, indent=2)
        with open(f'{OUT_DIR}/preds/dev_epoch{epoch:02d}.json', 'w', encoding='utf-8') as f:
            json.dump(dev_recs, f, ensure_ascii=False, indent=2)

    # 7. Test (best checkpoint)
    log.info('\n' + '=' * 96)
    log.info(f'训练结束。best epoch={best_epoch}, dev Macro-F1={best_f1*100:.2f}%')
    log.info('[Test] 载入 best checkpoint 评估 test 集...')
    load_ckpt_weights(f'{OUT_DIR}/best/ckpt.pth', sign_proj, adapter, llm, pose_enc, new_head)
    test_loader = make_loader(test_rows, 'test')
    tm, test_recs = evaluate(test_loader, minicpm, llm, pose_enc, adapter, sign_proj,
                             new_head, tok, base_vocab, 'test')
    log.info(fmt_metrics(tm, 'TEST (best checkpoint)',
                         f'  [best epoch={best_epoch}]'))
    t3 = sub_metrics_3class(test_recs)
    if t3:
        log.info(fmt_metrics(t3, 'TEST 补充诊断 — 三类 (system_speaking=0 子集)'))
    with open(f'{OUT_DIR}/preds/test_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(test_recs, f, ensure_ascii=False, indent=2)

    best_dev = history[best_epoch - 1]['dev'] if 0 < best_epoch <= len(history) else None
    total_h = (time.time() - t_start) / 3600
    with open(f'{OUT_DIR}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'exp': cfg['exp'], 'best_epoch': best_epoch,
                   'best_dev_macro_f1': round(best_f1, 6),
                   'dev_metrics_best': best_dev, 'test_metrics': tm,
                   'test_metrics_3class_sys0': t3,
                   'epoch_history': history, 'config': cfg,
                   'total_hours': round(total_h, 3)}, f, ensure_ascii=False, indent=2)
    write_report(cfg, history, best_epoch, best_f1, tm, t3, total_h)
    log.info(f'\n全部完成。耗时 {total_h:.2f} 小时。输出: {OUT_DIR}')


def write_report(cfg, history, best_epoch, best_f1, tm, t3, total_h):
    C = STATE_TOKENS
    L = ['# FullDuplex Sign Turn-Taking — Exp1 (GPU3)', '',
         '第一版四状态手语 turn-taking 训练 / 可学习性验证。', '',
         '## 1. 设置', '',
         f'- 数据: `{cfg["data_dir"]}` (train/dev/test.jsonl)',
         f'- 复用 checkpoint: `{cfg["reuse_checkpoint"]}` (Exp25 learned_down2, 架构未改)',
         f'- 模型输入: RGB `[0:window_end_frame]` ({cfg["rgb_frames"]} 帧 x 64 token, '
         f'MiniCPM-o 原生 SigLIP+resampler, 冻结) + Pose `[0:window_end_frame]` '
         f'(Uni-Sign→TemporalAdapter(/2)→Projector) + `system_speaking`',
         f'- 禁止入模: {", ".join("`%s`" % b for b in cfg["excluded_from_input"])}',
         f'- 新增 special tokens: {", ".join("`%s`" % t for t in NEW_TOKENS)}',
         f'- Assistant target: 仅 1 个状态 token, 其余位置 `labels=-100`',
         f'- Loss: {cfg["loss"]}',
         f'- train 对 `<WAIT>` {cfg["wait_oversample"]}x 加权过采样 (dev/test 不过采样, '
         f'原始 JSON 未修改)',
         f'- bf16, batch_size={cfg["batch_size"]}, grad_accum={cfg["grad_accum"]}, '
         f'epochs={cfg["num_epochs"]}',
         f'- best checkpoint 依据: **dev Macro-F1**', '',
         '### 可训练参数', '', '| 组 | 参数量 | lr |', '|---|---|---|']
    pc = cfg['param_counts']
    for k, nm in [('sign_projector', 'sign_projector'), ('temporal_adapter', 'temporal_adapter'),
                  ('lora', 'lora'), ('unisign_unfrozen', None), ('new_token_rows', 'new_token_rows')]:
        lr = cfg['lr'].get(nm, '—') if nm else \
            f'pose_proj {cfg["lr"]["pose_proj"]:.1e} / fusion_gcn {cfg["lr"]["fusion_gcn_modules"]:.1e}'
        lr = f'{lr:.1e}' if isinstance(lr, float) else lr
        L.append(f'| {k} | {pc[k]:,} | {lr} |')
    L += [f'| **合计** | **{pc["total_trainable"]:,}** | |',
          f'| 冻结 | {pc["frozen"]:,} | |',
          f'| 可训练占比 | {pc["trainable_ratio_pct"]}% | |', '',
          '## 2. 训练过程', '',
          '| Epoch | Train Loss | Dev Loss | Dev Acc | Dev Macro-F1 | Dev WAIT Recall | 时间 | 显存 |',
          '|---|---|---|---|---|---|---|---|']
    for h in history:
        d = h['dev']
        mark = ' **(best)**' if h['epoch'] == best_epoch else ''
        L.append(f'| {h["epoch"]}{mark} | {h["train_loss"]:.4f} | {d["loss"]:.4f} | '
                 f'{d["accuracy"]*100:.2f}% | {d["macro"]["f1"]*100:.2f}% | '
                 f'{d["wait_recall"]*100:.2f}% | {h["train_time_min"]:.0f}min | '
                 f'{h["peak_mem_gib"]}GiB |')
    L += ['', f'best epoch = **{best_epoch}** (dev Macro-F1 = **{best_f1*100:.2f}%**)', '',
          '## 3. Test 结果 (best checkpoint)', '',
          '| 指标 | 值 |', '|---|---|',
          f'| Accuracy | **{tm["accuracy"]*100:.2f}%** |',
          f'| Macro Precision | **{tm["macro"]["precision"]*100:.2f}%** |',
          f'| Macro Recall | **{tm["macro"]["recall"]*100:.2f}%** |',
          f'| Macro F1 | **{tm["macro"]["f1"]*100:.2f}%** |',
          f'| 样本数 | {tm["n"]} |', '',
          '### 各类 Precision / Recall / F1', '',
          '| 类别 | Precision | Recall | F1 | Support |', '|---|---|---|---|---|']
    for c in C:
        p = tm['per_class'][c]
        L.append(f'| `{c}` | {p["precision"]*100:.2f}% | {p["recall"]*100:.2f}% | '
                 f'{p["f1"]*100:.2f}% | {p["support"]} |')
    L += ['', '### 4x4 Confusion Matrix (行=真实, 列=预测)', '',
          '| 真实 \\ 预测 | ' + ' | '.join(f'`{c}`' for c in C) + ' |',
          '|---|' + '---|' * len(C)]
    for i, c in enumerate(C):
        L.append(f'| `{c}` | ' + ' | '.join(str(v) for v in tm['confusion_matrix'][i]) + ' |')
    ib = tm['incomplete_backchannel_confusion']
    L += ['', '### INCOMPLETE / BACKCHANNEL 混淆专项', '',
          f'- 真实 `<INCOMPLETE>` → 预测 `<BACKCHANNEL>`: '
          f'**{ib["INCOMPLETE_true_pred_BACKCHANNEL"]} / {ib["INCOMPLETE_support"]}** '
          f'({ib["rate_INCOMPLETE_to_BACKCHANNEL"]*100:.2f}%)',
          f'- 真实 `<BACKCHANNEL>` → 预测 `<INCOMPLETE>`: '
          f'**{ib["BACKCHANNEL_true_pred_INCOMPLETE"]} / {ib["BACKCHANNEL_support"]}** '
          f'({ib["rate_BACKCHANNEL_to_INCOMPLETE"]*100:.2f}%)',
          f'- 双向混淆合计: **{ib["total_IB_confusions"]}**', '',
          '### WAIT Recall', '',
          f'- **{tm["wait_recall"]*100:.2f}%** '
          f'({tm["per_class"]["<WAIT>"]["tp"]}/{tm["per_class"]["<WAIT>"]["support"]})', '',
          '> ⚠️ 本数据中 `system_speaking=1` 与 `<WAIT>` 在 train/dev/test 上 **完全一一对应** '
          '(train 56/56, dev 8/8, test 9/9), 且 `system_speaking` 是模型输入之一。'
          '因此 WAIT 可由该标志平凡解出, WAIT Recall 不构成有效的可学习性证据。']
    if t3:
        L += ['', '## 4. 补充诊断 — 三类 (system_speaking=0 子集)', '',
              '剔除可被 `system_speaking` 平凡解出的 WAIT 后, 真实可学习性体现在这里。', '',
              '| 指标 | 值 |', '|---|---|',
              f'| Accuracy | **{t3["accuracy"]*100:.2f}%** |',
              f'| Macro F1 | **{t3["macro"]["f1"]*100:.2f}%** |',
              f'| 样本数 | {t3["n"]} |', '',
              '| 类别 | Precision | Recall | F1 | Support |', '|---|---|---|---|---|']
        for c in t3['confusion_matrix_labels']:
            p = t3['per_class'][c]
            L.append(f'| `{c}` | {p["precision"]*100:.2f}% | {p["recall"]*100:.2f}% | '
                     f'{p["f1"]*100:.2f}% | {p["support"]} |')
        maj = max(t3['per_class'].items(), key=lambda kv: kv[1]['support'])
        L += ['', f'多数类基线 (全预测 `{maj[0]}`): Accuracy = '
                  f'{maj[1]["support"]/t3["n"]*100:.2f}%, Macro F1 = '
                  f'{2*(maj[1]["support"]/t3["n"])/(1+maj[1]["support"]/t3["n"])/3*100:.2f}%']
    L += ['', f'*总耗时 {total_h:.2f} 小时*']
    with open(f'{OUT_DIR}/final_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log.info(f'[Report] 已保存 {OUT_DIR}/final_report.md')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log.exception('训练异常终止')
        raise
