---
title: "Frog Blood Cell Morphology — Full Dataset Report"
date: "23 May 2026"
---

# Abstract

This report describes the **size and shape of frog blood cells** measured from microscopy
images. We focus on cells that passed an automatic quality check and were labelled **good**
for analysis. Cells labelled bad or rejected were not included in the size summaries.

The dataset covers **479 frogs** and **7,214 images**. In total,
**383,915** detected cells were scored; **57,520** (**15.0%**)
were accepted as good. All sizes below refer to this good-cell set only.

**Cell size.** The typical (median) cell covers **290.69 $\mu$m$^2$** of area. Half of
good cells fall between **257.70** and **335.48 $\mu$m$^2$** (the middle 50% of the
distribution). That area is similar to a **circle about 19.24 $\mu$m wide**
(a simple way to picture area as a length; the cell is not necessarily round). The average cell
area was **303.10 $\pm$ 67.07 $\mu$m$^2$** (mean $\pm$ SD across all good cells).

**Nucleus size.** For **57,218** cells we also measured the nucleus. The
N/C ratio is nucleus area divided by cell area (a number between 0 and 1). The typical value
was **0.13**; the mean $\pm$ SD across cells was **0.14 $\pm$ 0.04**.

**Differences between frogs.** Cell size was **not the same in every frog**. Each frog has an
average cell area; those averages ranged from **212.55** to
**482.23 $\mu$m$^2$**. The **typical frog’s average** (median across frogs, each frog
counted once) was **290.34 $\mu$m$^2$**. Roughly **0.661** of the variation
in cell area between cells is due to **which frog** the cell came from. The rest is how much
cells differ **within the same frog** — typical spread about **11.7%**
of that frog’s mean area (e.g. $\approx\pm$ 12% around a frog’s average). In short: frogs differ in
average cell size, but cells from one frog are usually similar to each other.

**Nucleus vs cell size.** When cells get larger, the nucleus does not grow as fast as the cell
body. Larger cells therefore tend to have a **proportionally smaller nucleus** (a lower N/C
ratio).

# Methods

**Images and cohort.** We analysed microscopy images of **frog blood cells** from
**479 frogs** (**7,214 images**). Each detected cell was linked to a
**frog ID** parsed from the image file name. All size measurements are reported in
**micrometres ($\mu$m)** or **square micrometres ($\mu$m$^2$)** after converting from pixels.

**Segmentation.** Cells were segmented automatically from each image using **Cellpose**
(membrane/nucleus masks).
Each connected region in the cell mask is treated as one candidate cell.

**Quality classification.** Every cell crop was scored by a trained **ResNet18**
classifier, which outputs **p(good)** — the model’s estimated probability that the
cell quality is good enough to measure. We used **selective rejection** with three
outcomes:

- **p(good) $\leq$ 0.10** $\rightarrow$ **bad** (poor quality; excluded from size analysis)
- **p(good) $\geq$ 0.76** $\rightarrow$ **good** (good quality; included in all morphometry and summaries)
- **0.10 < p(good) < 0.76** $\rightarrow$ **rejected** (model uncertain; excluded from size analysis)

Of **383,915** scored detections, **57,520** (**15.0%**)
were labelled good, **5,535** were rejected as uncertain, and the remainder
were bad. **Only good cells** enter the size results in this report.

**Morphometry.** For each good cell we measured:

- **Cell area** ($\mu$m$^2$) and **long / short diameter** ($\mu$m)
- **Nucleus area and axes**, **N/C ratio** (nucleus area $\div$
  cell area), and **axis ratio** (major $\div$ minor axis)

Pixel measurements were converted to $\mu$m using the **pixel size read from each image’s
metadata** (OME-TIFF / TIFF), when available.

**Aggregation and yield.** For each frog we computed the **mean and SD** of cell area
(and other metrics) across its good cells. **Yield** (good fraction) is the share of
scored detections labelled good, reported per frog and per image to show how much of
each sample passed the quality filter.

**Statistical analysis.**

- **Reference intervals:** 2.5th, 50th (median), and 97.5th percentiles of good-cell
  measurements
- **Between-frog variation:** ICC — the share of cell-area variation explained by
  **which frog** a cell came from
- **Within-frog spread:** for each frog, CV = (SD $\div$ mean) $\times$ 100%; we report the median
  across frogs
- **Nucleus--cell scaling:** log--log regression of nucleus size vs cell size; slope $\approx$ 1
  means nucleus and cell grow together; slope < 1 means the nucleus grows more slowly
- **Mixed-effects models:** the same regressions with a **random intercept per frog**, so
  many cells from the same frog are not treated as independent repeats
- **N/C vs cell size:** mixed model for log(N/C) vs log(cell area)

# Cell area distribution

Distribution of cell areas ($\mu$m$^2$) for all good cells.

![Cell area distribution](./figures/AreaDistribution.png)

- **Median cell area: 290.69 $\mu$m$^2$** (IQR 257.70--335.48 $\mu$m$^2$).
- **Mean $\pm$ SD: 303.10 $\pm$ 67.07 $\mu$m$^2$.**
- **Circle equivalent: 19.24 $\mu$m** — the diameter of a circle with the same area as the median cell ($d = 2\sqrt{A/\pi}$).

# Cell diameter distribution

Long-axis and short-axis diameters ($\mu$m) for good cells. The two
distributions track each other closely.

![Cell diameter distribution](./figures/DiameterDistribution.png)

# Nucleus size distribution

![Nucleus distribution](./figures/NucleusDistribution.png)

# Nucleus-to-cell area ratio (N/C)

Distribution of nucleus area divided by cell area.

![N/C ratio distribution](./figures/NCRatioDistribution.png)

- **Median N/C ratio: 0.13** (nucleus area $\approx$ 0.13 of cell area — about 13% of the cell).
- **Mean $\pm$ SD: 0.14 $\pm$ 0.04.**

# Per-frog variation in cell and nucleus area

Boxplots for the top-40 frogs by good-cell count, sorted by per-frog median cell area.

![Per-frog cell area](./figures/PerFrogBoxplot.png)

![Per-frog nucleus area](./figures/PerFrogNucleusBoxplot.png)

Per-frog mean cell area spans **212.55--482.23 $\mu$m$^2$**.
The typical frog’s average (median across frogs) was **290.34 $\mu$m$^2$**.

# Classification yield

Not every detected cell is measured in this report. After segmentation, the quality
classifier labels each cell **good**, **bad**, or **rejected** (uncertain). **Yield** is
the **good fraction** — the share of scored detections that were accepted as good enough
to measure. This section shows how that fraction varies across frogs and images.

**What to expect in the figure below.** The **left panel** is a stacked bar chart for the
first 80 frogs: **green** = good cells, **orange** = bad, **pink** = rejected. Taller green
stacks mean more cells passed the filter for that frog. The **right panel** plots each frog’s
good fraction against its total good-cell count — it shows whether frogs with more accepted
cells also tend to have higher (or lower) yield. The summary numbers after the figure give
cohort-wide yield and the per-frog / per-image range.

![Classification yield](./figures/YieldByFrog.png)

Of **383,915** scored detections, **57,520** were accepted as good
(**15.0%**). Per-frog good fraction: median **17.7%**
(range **2.7--62.1%**). Per-image median good fraction:
**17.6%**; **16** images had zero good cells.

**Important.** All cell-size results in this report use **good cells only**. Cells labelled
bad or rejected are **not** included in area, diameter, or N/C summaries. When you read those
size numbers, also check **yield**: if a frog or image had a **low good fraction**, fewer cells
were measured there, so its size average may be less reliable and the sample may not represent
everything that was seen in the microscope image.

# Nucleus--cell scaling (regression)

This section asks: **when a cell gets larger, does the nucleus grow in proportion?** We fit
**log--log regressions** on good cells with a matched nucleus (57,218 cells, 479 frogs). On a
log--log plot, a straight line means nucleus size scales as a **power** of cell size. A slope
of **1** is isometric scaling (nucleus and cell grow at the same rate); **below 1** is
negative allometry (the nucleus grows more slowly than the cell).

**What to expect in the figures below.** There are **three scatter plots**, one for each size
pair:

1. **Nucleus area vs cell area**
2. **Nucleus long axis vs cell long axis**
3. **Nucleus short axis vs cell short axis**

Each plot uses **log--log axes** (both axes logarithmic). Every point is one cell; the top
frogs by cell count are shown in **colour**, the rest in grey. Each plot overlays a
**solid black OLS line** (one best-fit line through all pooled cells) and a **dashed orange
Mixed line** (the within-frog trend with a random intercept per frog).

![Nucleus area vs cell area](./figures/NucleusVsCellRegression.png)

![Nucleus major vs cell major axis](./figures/NucleusMajorVsCellMajor.png)

![Nucleus minor vs cell minor axis](./figures/NucleusMinorVsCellMinor.png)

**What to expect in the table below.** One row per comparison (area, long axis, short axis).
The table gives the numeric slopes that match the plots:

- **OLS slope** — ordinary least squares across **all cells** (each cell counted separately;
  mixes variation between and within frogs).
- **Mixed slope** — mixed-effects model with a **random intercept per frog**; estimates how
  nucleus size changes with cell size **among cells from the same frog**.
- **OLS 95% CI** — uncertainty interval for the OLS slope.
- **R$^2$** — fraction of variance in log(nucleus) explained by log(cell) in the **OLS** model
  (does not apply to the mixed model).
- **Slope < 1** in every row — negative allometry: larger cells have proportionally smaller
  nuclei along that measure. Mixed slopes are **smaller** than OLS because much of the
  nucleus--cell correlation across the cohort is **between frogs**, not within them.

| Comparison | n | OLS slope | OLS 95% CI | Mixed slope | R$^2$ |
| --- | --- | --- | --- | --- | --- |
| Nucleus area vs cell area | 57,218 | 0.526 | [0.518, 0.533] | 0.149 | 0.255 |
| Nucleus long axis vs cell long axis | 57,218 | 0.616 | [0.609, 0.624] | 0.368 | 0.311 |
| Nucleus short axis vs cell short axis | 57,218 | 0.426 | [0.418, 0.434] | 0.182 | 0.148 |

: **Table — Nucleus--cell scaling regressions** (good cells with matched nucleus; 57,218 cells, 479 frogs). OLS = fit across all cells; Mixed = random intercept per frog. R$^2$ = fraction of variance in log(nucleus) explained by log(cell) in the OLS model.

# N/C ratio vs cell size

**N/C ratio vs cell size.** The plot below shows whether larger cells have a proportionally
smaller or larger nucleus (lower or higher N/C). The table summarizes mixed-effects models
on **log(N/C) ~ log(cell area)**.

**Random intercept per frog** — each frog may have a different baseline N/C; we test whether
N/C still changes with cell area across cells. **Random intercept + slope per frog** —
allows the N/C--size trend to vary by frog as well. A **negative slope** means larger cells
tend to have **lower N/C** (a proportionally smaller nucleus). Both models show a clear
negative slope.

![N/C vs cell area](./figures/NCRatioVsArea.png)

| Model | Slope (log cell area) | 95% CI | n | Frogs |
| --- | --- | --- | --- | --- |
| Random intercept per frog | -0.8515 | [-0.8597, -0.8433] | 57,218 | 479 |
| Random intercept + slope per frog | -0.8134 | [-0.8342, -0.7926] | 57,218 | 479 |

: **Table — Mixed models for N/C vs cell area** (log--log; good cells with matched nucleus). Slope = change in log(N/C) per unit change in log(cell area).

# Reference intervals

Normative percentile ranges for good-cell morphology:

| Metric | 2.5th %ile | Median | 97.5th %ile |
| --- | --- | --- | --- |
| Cell area (µm²) | 200.90 | 290.69 | 463.10 |
| Cell equivalent diameter (µm) | 15.99 | 19.24 | 24.28 |
| Cell long axis (µm) | 19.40 | 24.52 | 32.57 |
| Cell short axis (µm) | 12.15 | 15.23 | 19.36 |
| Nucleus area (µm²) | 28.57 | 39.06 | 63.36 |
| N/C area ratio | 0.09 | 0.13 | 0.21 |
| Cell axis ratio | 1.25 | 1.62 | 2.08 |

# Frog summary snapshot

Ten frogs with the **smallest** mean cell area and ten with the **largest**, plus a rank scatter of outlier frogs.

![Outlier frogs](./figures/OutlierFrogs.png)

Ten frogs with the **smallest** mean cell area:

| frog_id | n_cells | area_um2_mean | nc_ratio_mean | good_fraction |
| --- | --- | --- | --- | --- |
| 89.0 | 119.0 | 212.55 | 0.16 | 0.19 |
| 41.0 | 187.0 | 216.02 | 0.19 | 0.18 |
| 101.0 | 187.0 | 218.23 | 0.18 | 0.18 |
| 35.0 | 138.0 | 219.15 | 0.14 | 0.21 |
| 99.0 | 146.0 | 221.54 | 0.18 | 0.17 |
| 73.0 | 114.0 | 221.88 | 0.17 | 0.12 |
| 81.0 | 199.0 | 227.21 | 0.15 | 0.26 |
| 30.0 | 127.0 | 227.88 | 0.16 | 0.24 |
| 28.0 | 100.0 | 228.04 | 0.18 | 0.13 |
| 250.0 | 168.0 | 229.04 | 0.15 | 0.21 |

Ten frogs with the **largest** mean cell area:

| frog_id | n_cells | area_um2_mean | nc_ratio_mean | good_fraction |
| --- | --- | --- | --- | --- |
| 221.0 | 89.0 | 449.54 | 0.11 | 0.09 |
| 218.0 | 78.0 | 450.36 | 0.12 | 0.06 |
| 210.0 | 116.0 | 454.62 | 0.12 | 0.23 |
| 166.0 | 98.0 | 456.92 | 0.12 | 0.23 |
| 431.0 | 99.0 | 457.59 | 0.16 | 0.13 |
| 154.0 | 98.0 | 462.55 | 0.11 | 0.15 |
| 455.0 | 83.0 | 465.68 | 0.11 | 0.48 |
| 443.0 | 69.0 | 467.09 | 0.11 | 0.05 |
| 119.0 | 90.0 | 469.63 | 0.11 | 0.08 |
| 440.0 | 136.0 | 482.23 | 0.15 | 0.21 |

# Discussion

This report describes the size and shape of **good-quality frog blood cells** from a
large automated microscopy cohort. Across **57,520** accepted cells from
**479** frogs, the typical cell covered about **290.69 $\mu$m$^2$**
(median area; middle 50% between **257.70** and **335.48 $\mu$m$^2$**),
roughly the area of a circle **19.24 $\mu$m** wide. That picture
is useful as a **reference description** of well-segmented, classifier-approved cells in
this dataset.

A striking feature is **how much average cell size differs between frogs**. Per-frog mean
areas ranged from **212.55** to **482.23 $\mu$m$^2$**,
while the typical frog's average (median across frogs) was **290.34 $\mu$m$^2$**.
Roughly **0.661** of the total variation in cell area is associated with **which frog**
a cell came from; the rest reflects spread **within** each frog (median within-frog CV
about **11.7%**). That pattern may reflect real biological
differences between frogs, differences in sample preparation or imaging, or both; we
cannot separate those sources here. Frogs at the extremes of the size rank (see the frog
summary snapshot and outlier plot) are **candidates for manual review**: they may represent
genuine tail biology, technical artefacts, or low cell counts that make the per-frog average
less stable.

**Nucleus size relative to cell size** also carries a consistent message. Nucleus--cell
scaling is negatively allometric (nucleus grows slower than cell area; larger cells are relatively less nucleus-dense). In plain terms, when cells get larger, the nucleus
tends to grow **more slowly** than the cell body, so larger cells often show a **lower
N/C ratio** (a smaller nuclear share of cell area). That is **consistent with** a model in
which cytoplasm expands faster than nuclear content during cell enlargement, though we
have not tested underlying mechanisms. The mixed-effects regressions suggest that part of
the nucleus--cell correlation across the cohort reflects **between-frog** differences, not
only within-frog co-variation---so population-level trends should be read alongside
per-frog summaries.

**Classifier yield** shapes every morphometric statement in this report. Only
**15.0%** of **383,915** scored detections were labelled good
(median per-frog good fraction **17.7%**, range **2.7--62.1%**).
Cells with uncertain quality were **rejected** rather than forced into good or bad classes.
Morphometry therefore describes the **accepted subset**---cells the model judged clear
enough to measure---not the full pool of segmented objects. Frogs or images with **low yield**
contributed fewer measured cells; **16** images had no good cells at all.
When interpreting a frog's mean area or N/C, it is worth checking its good fraction in the
exported summary tables: a sparse or selective accepted set may not represent everything
visible in the image.

# Appendix

Full per-frog metrics (all morphometry columns, yield, and flags) are exported as
`frog_summary_report.csv` and `frog_summary_report.xlsx`.

# Glossary

| Term | Meaning |
|------|---------|
| **Good cell** | A segmented cell accepted by the quality classifier for morphometry. |
| **N/C ratio** | Nucleus area divided by cell area (unitless). |
| **Yield / good fraction** | Fraction of scored detections accepted as good for a frog or image. |
| **ICC** | Share of cell-area variation explained by differences between frogs (which frog a cell came from). |
| **Within-frog spread (CV)** | For each frog, (SD of cell areas $\div$ mean area) $\times$ 100%; report gives the median across frogs. |
| **Typical frog’s average** | Median of per-frog mean cell areas — each frog counts once, not weighted by cell count. |
| **Circle equivalent diameter** | Diameter of a circle with the same area as the cell; a readable length summary, not the cell’s true width. |
| **Allometry** | How nucleus size scales with cell size; slope = 1 is isometric, < 1 is negative allometry. |
| **Reference interval** | Percentile range (e.g. 2.5th--97.5th) describing typical good-cell morphology. |