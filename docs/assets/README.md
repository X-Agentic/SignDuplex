# Paper homepage artwork

The current homepage uses self-contained SVG vector artwork. Logo geometry and chart labels remain sharp when zoomed; all diagram text is converted to paths for consistent rendering. Full-resolution PNG fallbacks are also included at 3× scale.

## Official institution marks

- `tsinghua-emblem.svg`: the original emblem vector path from page 4 of [Tsinghua University's identity manual](https://www.env.tsinghua.edu.cn/qinghuadaxueerjidanweibiaozhiguifanshouce-shixing.pdf). Proportions and source color are preserved.
- `huawei-logo.svg`: the full vertical flower and wordmark from [Huawei's official SVG asset](https://www.huawei.com/-/media/htemplate-home/1.0.1.20251205144752/components/assets/img/favicon-logo.svg). Class styling is duplicated inline for renderer compatibility; mark geometry and colors are unchanged.
- `brand-banner.svg`: a large bilingual institution lockup composed from those two marks. `brand-banner-3x.png` is a 3900×960 fallback. The marks identify the paper authors' affiliations.

## Paper visualizations

- `method-overview.svg`: a readable diagram of the proposed method in Section 3.2 and Figure 3.
- `results-quality.svg`, `results-latency.svg`: separate zero-based charts transcribing the continuous-input values in Table 4, with the original-boundary reference labelled separately.
- Each diagram has an editable text SVG and a 3120-pixel-wide PNG export. Native README tables preserve the exact paper values as accessible text.

Rebuild the technical figures with `python docs/assets/build_paper_visuals.py`. The banner script requires an installed Noto Sans CJK SC font and CairoSVG; pass the font file to `python docs/assets/build_brand_banner.py /path/to/NotoSansCJKsc-Regular.otf`. The font is available from the [Noto CJK project](https://github.com/notofonts/noto-cjk) under the SIL Open Font License.

Design dependencies are separate from model dependencies. Archived images from the preceding homepage remain in this directory for reproducibility but are not displayed by the current homepage.
