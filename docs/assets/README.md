# Paper homepage artwork

The current homepage uses self-contained SVG vector artwork. Logo geometry and chart labels remain sharp when zoomed; all diagram text is converted to paths for consistent rendering. Full-resolution PNG fallbacks are also included at 3× scale.

## Official institution marks

- `tsinghua-emblem.svg`: the original emblem vector path from page 4 of [Tsinghua University's identity manual](https://www.env.tsinghua.edu.cn/qinghuadaxueerjidanweibiaozhiguifanshouce-shixing.pdf). Proportions and source color are preserved.
- `tsinghua-calligraphy.svg`: the actual four-character brush-calligraphy vector from the cover of the same official manual. Its geometry is preserved and its white reverse-version fill is set to the emblem's purple for the white header.
- `huawei-logo.svg`: the full vertical flower and wordmark from [Huawei's official SVG asset](https://www.huawei.com/-/media/htemplate-home/1.0.1.20251205144752/components/assets/img/favicon-logo.svg). Class styling is duplicated inline for renderer compatibility; mark geometry and colors are unchanged.
- `huawei-horizontal.svg`: the same official flower and lettering paths arranged horizontally, with uniform scaling within each component. The reference is the horizontal corporate logo in [Huawei's official partner branding manual](https://e.huawei.com/marketingcloud/pep/asset/20000001/Material/45de3cd25cc4490ca407764f37c3a262/M3T1A669N1101884873329614975/03_%20Partner%20Co-Branding%20Guidelines%20V1_0.pdf).
- `brand-banner.svg`: a 1400×420 white institution header following the supplied reference: official Tsinghua emblem and calligraphy, one-line English caption, gray multiplication sign, horizontal Huawei mark, and a centered SignDuplex label between purple and red rules. `brand-banner-3x.png` is a 4200×1260 fallback. The marks identify the paper authors' affiliations.
- `institutions-banner.svg`: the same upper institution vector elements cropped to 1300×260 for the website, with the bottom SignDuplex label and rules removed. Original geometry, colors and proportions are preserved. `institutions-banner.png` is its 3900×780 fallback. `institutions-provenance.json` records the source and crop. The existing README header remains unchanged.

## Paper visualizations

- `method-summary.svg`: a compact 1000×200 overview of causal RGB, pose and state input, Qwen3-0.6B with LoRA, five-event prediction, and interaction control. `method-summary.png` is its 3000×600 fallback.
- `method-overview.svg`: a readable diagram of the proposed method in Section 3.2 and Figure 3.
- `results-quality.svg`, `results-latency.svg`: separate zero-based charts transcribing the continuous-input values in Table 4, with the original-boundary reference labelled separately.
- The detailed diagram and charts have editable text SVGs and 3120-pixel-wide PNG exports. Native README tables preserve the exact paper values as accessible text.

The checked-in illustrations support the project introduction. Source attribution and derivations are recorded in `brand-provenance.json` and `institutions-provenance.json`.
