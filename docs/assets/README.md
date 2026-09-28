# Homepage visual assets

The README uses local PNG assets so its layout works with GitHub's sanitized Markdown, without JavaScript, external fonts, or third-party badge services. Editable SVG sources are included.

## Sources

- `tsinghua-emblem.jpg`: unchanged artwork downloaded from [Tsinghua's visual identity website](https://vi.tsinghua.edu.cn/gk/xxbz/xh/bzjbzsy.htm), [original image](https://vi.tsinghua.edu.cn/__local/8/1C/21/2F02A66E31BF084FE340463BB47_B9A95F73_185BC.jpg). The hero uses a viewport to omit the original white margins; the emblem itself is unchanged.
- `huawei-logo.png`: unchanged horizontal logo from the [Huawei website](https://www.huawei.com/en), [original image](https://www.huawei.com/-/media/hcomponent-header/1.0.1.20260908162100/component/img/huawei_logo.png).
- `paper-architecture.png`: Figure 3 rendered directly from page 3 of the [supplied final manuscript](../paper/SignDuplex.pdf), without changing the figure's contents.
- `hero`, `interaction`, and `results`: project-specific layouts. Method and result values follow the supplied final manuscript, especially Sections 3–4 and Tables 1–4. The state diagrams are schematic, not recorded demo footage.

Institution marks identify the authors' affiliations. They remain the property of their respective institutions. Logo downloads were checked on 2026-09-28.

## Regenerate the artwork

In a separate design environment with Python, CairoSVG, and the DejaVu Sans font installed:

```bash
python docs/assets/build_visuals.py
```

This regenerates the three SVG/PNG pairs from the local original logo assets. Design dependencies are not required to run SignDuplex. Keep the text alternatives and native tables in the README synchronized with any changes to the figures.
