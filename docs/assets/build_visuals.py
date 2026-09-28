"""Build README artwork. Run with Python and CairoSVG; no model dependencies."""
from pathlib import Path
import base64
import html

ROOT = Path(__file__).resolve().parent
PURPLE, RED, INK, MUTED = '#702082', '#cf243b', '#20162c', '#706879'

def text(x, y, value, size=24, color=INK, weight=400, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'

def rect(x,y,w,h,fill='#fff',stroke='none',r=20):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>'

def line(x1,y1,x2,y2,color='#e4dce9',width=2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="{width}" fill="none"/>'

def arrow(x1,y1,x2,y2,color=PURPLE):
    return line(x1,y1,x2,y2,color,3)+f'<path d="M{x2-9} {y2-6} L{x2} {y2} L{x2-9} {y2+6}" fill="none" stroke="{color}" stroke-width="3"/>'

def raster(name,x,y,w,h):
    mime='image/jpeg' if name.endswith('.jpg') else 'image/png'
    data=base64.b64encode((ROOT/name).read_bytes()).decode()
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:{mime};base64,{data}"/>'

def save(name,h,parts,title):
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{h}" viewBox="0 0 1600 {h}" role="img" aria-labelledby="title"><title id="title">{html.escape(title)}</title><defs><linearGradient id="wash" x2="1" y2="1"><stop stop-color="#fbf8fd"/><stop offset="1" stop-color="#fff7f7"/></linearGradient><linearGradient id="accent"><stop stop-color="{PURPLE}"/><stop offset="1" stop-color="{RED}"/></linearGradient></defs><g font-family="DejaVu Sans, Arial, sans-serif">{''.join(parts)}</g></svg>'''
    (ROOT/f'{name}.svg').write_text(svg)
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(ROOT/f'{name}.png'))

# Hero: institution marks sit on white, with the original logo artwork intact.
p=[rect(0,0,1600,760,'#fff',r=0)]
p += [f'<svg x="62" y="28" width="84" height="84" viewBox="460 70 460 460">{raster("tsinghua-emblem.jpg",0,0,1400,600)}</svg>',text(165,68,'TSINGHUA',24,PURPLE,700),text(165,97,'UNIVERSITY',18,MUTED),text(374,82,'×',32,'#aea4b5'),raster('huawei-logo.png',428,47,222,50),text(1536,78,'SIGN LANGUAGE  /  FULL DUPLEX',18,MUTED,500,'end'),line(64,140,1536,140)]
p += [rect(0,166,1600,594,'url(#wash)',r=28),rect(64,216,228,38,'#eee3f3',r=19),text(178,242,'RESEARCH PROJECT',16,PURPLE,700,'middle'),text(59,378,'SignDuplex',110,INK,700),text(66,443,'See the signing.',37,INK,500),text(66,493,'Know when to respond.',37,PURPLE,500),text(66,558,'Real-time interaction control for',24,MUTED),text(66,594,'continuous sign language communication.',24,MUTED)]
# Schematic: bidirectional visual perception and response remain active.
p += [rect(966,235,564,386,'#fff','#e9dfea',24),text(1248,282,'CONTINUOUS VISUAL PERCEPTION',17,MUTED,600,'middle')]
p += [f'<path d="M1043 370 C1060 284 1414 284 1440 370" fill="none" stroke="{PURPLE}" stroke-width="3"/>',f'<path d="M1043 458 C1060 552 1414 552 1440 458" fill="none" stroke="{RED}" stroke-width="3"/>',text(1245,329,'COMPLETE →',20,PURPLE,600,'middle'),text(1245,557,'← BARGE',20,RED,600,'middle'),rect(995,365,222,97,'#f3ebf7',r=16),rect(1273,365,228,97,'#fff0f1',r=16),text(1106,405,'RECEIVING',23,PURPLE,700,'middle'),text(1106,438,'User is signing',18,MUTED,400,'middle'),text(1387,405,'RESPONDING',23,RED,700,'middle'),text(1387,438,'System is speaking',18,MUTED,400,'middle'),text(1248,594,'Two states. Five events. One controller.',20,MUTED,400,'middle')]
p += [line(64,653,1536,653),text(64,711,'RGB + POSE',20,PURPLE,700),text(389,711,'CAUSAL INPUT',20,PURPLE,700),text(793,711,'5 INTERACTION EVENTS',20,PURPLE,700),text(1281,711,'PLUG & PLAY',20,PURPLE,700)]
save('hero',760,p,'SignDuplex — Tsinghua University × Huawei. Real-time full-duplex sign language interaction.')

# One readable explanatory graphic, with native README text supplying the details.
p=[rect(0,0,1600,716,'#fcfafd',r=24),text(56,65,'FROM CONTINUOUS SIGNING TO TIMELY RESPONSE',20,PURPLE,700),text(56,116,'Interaction timing, independently modeled.',36,INK,600)]
for x,n,title,desc in [(56,'01','Perceive','Causal video prefix'),(567,'02','Decide','State-aware event prediction'),(1078,'03','Act','Route to downstream tasks')]:
    p += [text(x,181,n,20,PURPLE,700),text(x+46,183,title,27,INK,600),text(x,222,desc,21,MUTED)]
p += [rect(56,252,410,96,'#efe7f5'),text(82,291,'RGB pathway',24,PURPLE,600),text(82,325,'Appearance · face · gaze',21,MUTED),rect(56,366,410,96,'#eaf3f5'),text(82,405,'Pose pathway',24,'#286571',600),text(82,439,'Hands · motion · temporal cues',21,MUTED),text(56,510,'+ current system state',23,MUTED)]
p += [arrow(477,355,553,355),rect(567,252,430,256,'#fff','#ddcce7'),text(782,303,'SignDuplex',36,PURPLE,700,'middle'),text(782,345,'RGB + Pose + State tokens',21,MUTED,400,'middle'),line(600,368,964,368),text(782,410,'Qwen3-0.6B · LoRA',24,INK,500,'middle'),text(782,465,'Unified five-class event head',21,MUTED,400,'middle'),arrow(1008,355,1064,355)]
for y,title,desc,color,bg in [(252,'KEEP RECEIVING','REJECT · INCOMPLETE · WAIT',PURPLE,'#f0e8f6'),(354,'GENERATE RESPONSE','COMPLETE → SLT → LLM → TTS','#256256','#ebf5ef'),(456,'STOP OUTPUT','BARGE → return to RECEIVING',RED,'#fff0f1')]:
    p += [rect(1078,y,466,88,bg),text(1102,y+34,title,21,color,700),text(1102,y+66,desc,19,MUTED)]
p += [line(56,581,1544,581),text(56,626,'EVERY 1 SECOND',18,PURPLE,700),text(357,626,'Observe → predict → route. Restart input accumulation on state transition.',21,MUTED),text(56,675,'Visual perception continues while the system responds, enabling user interruption.',24,INK,500)]
save('interaction',716,p,'Paper method: causal RGB and pose prefixes plus state tokens predict five events; COMPLETE generates a response and BARGE stops output.')

# Linear axes start at zero; values are copied from Table 4 of the final manuscript.
p=[rect(0,0,1600,668,'#fcfafd',r=24),text(56,65,'CONTINUOUS MULTI-SENTENCE TRANSLATION',20,PURPLE,700),text(56,115,'Better translations. Earlier first submission.',36,INK,600)]
for x,title,sub in [(56,'Translation quality','BLEU-4 · higher is better'),(840,'First submission','Seconds · lower is better')]:
    p += [text(x,184,title,28,INK,600),text(x,221,sub,21,MUTED)]
rows=[('Offline',8.20,15.11),('SL Segmentation',11.85,6.48),('Online CSLR/SLT',10.90,6.26),('SignDuplex',13.72,5.99)]
for i,(name,bleu,lat) in enumerate(rows):
    y=279+i*65; color=PURPLE if i==3 else '#c5bdce'
    p += [text(56,y+23,name,20,INK,700 if i==3 else 400),rect(282,y,bleu/16*375,32,color,r=5),text(282+bleu/16*375+12,y+24,f'{bleu:.2f}',21,PURPLE if i==3 else MUTED,600)]
    color=RED if i==3 else '#c5bdce'
    p += [text(840,y+23,name,20,INK,700 if i==3 else 400),rect(1066,y,lat/16*365,32,color,r=5),text(1066+lat/16*365+12,y+24,f'{lat:.2f}',21,RED if i==3 else MUTED,600)]
for x,w in [(282,375),(1066,365)]:
    p += [line(x,527,x+w,527)]
    for n in [0,4,8,12,16]:p += [text(x+n/16*w,553,str(n),16,MUTED,400,'middle')]
p += [line(56,585,1544,585),text(56,630,'+67.3% BLEU-4',27,PURPLE,700),text(440,630,'−60.4% first submission time',27,RED,700),text(1544,630,'vs. whole-video offline · Table 4',19,MUTED,400,'end')]
save('results',668,p,'Paper Table 4: continuous-input BLEU-4 improves from 8.20 to 13.72; first submission decreases from 15.11 to 5.99 seconds.')
