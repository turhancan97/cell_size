# Cell Classifier and Cell Size Report

## Summary

We trained a small deep-learning model (ResNet18 backbone) that looks at
every segmented cell and decides whether it is a **good** cell worth
measuring or a **bad** segmentation that should be thrown away. On an
independent test set of **1,300 cells** the model already does very
well: it correctly accepts good cells **~95% of the time** and correctly
discards bad cells **~99% of the time**, with overall accuracy
**~98%**. If we additionally let the model say "I'm not sure, please
don't count this cell" on the most ambiguous crops, the cells it does
classify reach **~99% accuracy**, while only ~2% of cells are deferred
for human review.

Applying this filter to the full imaging dataset keeps 2,566
high-confidence good cells from 277 frogs for size measurements.
Those cells have a **median cell area of ~293 µm²**, a **median
nucleus-to-cell ratio of ~0.14**, and meaningful variability between
individual frogs — the per-frog mean cell area spans ~206 µm² to
540 µm². A log-log regression also shows that the nucleus is
**negatively allometric** with the cell: larger cells do have larger
nuclei, but the nucleus grows much slower than the cell, so big cells
are relatively less nucleus-dense.

## What the numbers mean

The report uses standard machine-learning terminology. Each term is
defined here once, in plain language:

| Term | Meaning in this report |
|---|---|
| **Positive class** | a **good** cell (one we want to keep and measure). |
| **Negative class** | a **bad** cell (we want to discard). |
| **Accuracy** | out of all cells, what fraction did the model label correctly. |
| **Recall** | out of the cells a human reviewer agreed are good, what fraction did the model also keep. High recall = few missed good cells. |
| **Precision** | out of the cells the model called good, what fraction actually are good. High precision = few bad cells slipping through. |
| **F1** | a single summary score that combines precision and recall (their harmonic mean). Closer to 100% is better. |
| **Specificity** | out of the cells the model called bad, what fraction actually are bad. High specificity = few bad cells leaking through as "good". |
| **True Positive (TP)** | good cell, kept. |
| **True Negative (TN)** | bad cell, discarded. |
| **False Positive (FP)** | bad cell that was kept by mistake. |
| **False Negative (FN)** | good cell that was discarded by mistake. |
| **Confusion matrix** | a 2×2 table of those four counts. |
| **Selective rejection** | an option to let the model say "I'm not sure" instead of forcing a yes/no for every cell. |
| **Coverage** | the fraction of cells the model was confident enough to label (the rest become "rejected/unsure"). |
| **Calibration / temperature scaling** | a post-hoc rescaling of the model's confidence values so that "80% confident" really does correspond to being right ~80% of the time. |

## Part A — Baseline classifier (single decision threshold)

### How it decides

The classifier outputs a probability `p(good)` between 0 and 1 for
every cell. In **baseline mode** we keep the cell if `p(good) ≥ 0.50`
and discard it otherwise. No cells are deferred.

### Headline metrics on the test split

| Metric | Value |
|---|---|
| **Accuracy** | **98.08%** |
| **Recall** (sensitivity to good cells) | **94.71%** |
| **Precision** | **92.27%** |
| **F1** | **93.47%** |
| **Specificity** (correctly discarded bad cells) | **98.65%** |

### Confusion matrix — raw counts

| True \ Pred | Pred `bad` | Pred `good` | Row total |
|---|---|---|---|
| True `bad`  | **1096** (TN) | **15** (FP) | 1111 |
| True `good` | **10** (FN)   | **179** (TP) | 189 |

### Confusion matrix — row-normalized (%)

| True \ Pred | Pred `bad` | Pred `good` |
|---|---|---|
| True `bad`  | 98.65% | 1.35% |
| True `good` | 5.29%  | 94.71% |

**Interpretation in plain language.** The model is now very strong on
both classes. About 99 out of every 100 mis-segmented cells are
correctly rejected, and about 95 out of every 100 genuinely good cells
are kept. When the model says a cell is good, it is right about **92%**
of the time — the remaining 8% are false alarms, which are typically
crops that *look* like a clean cell but were judged imperfect by the
human reviewer.

### Qualitative examples — what the model is seeing

Each panel shows up to 20 randomly sampled test cells from that
category, with the model's `p(good)` printed above each cell.

**Correctly kept good cells (true positives):**
![True Positive examples](./figures/TruePositive.png)

**Correctly discarded bad cells (true negatives):**
![True Negative examples](./figures/TrueNegative.png)

**Bad cells kept by mistake (false positives):** these are the
segmentations that leaked through the filter. Common failure modes are
blurred membranes, nearby touching cells, or incomplete outlines.
![False Positive examples](./figures/FalsePositive.png)

**Good cells rejected by mistake (false negatives):** the cells we
lose. They tend to be slightly atypical morphologies — elongated,
partially overlapping, or near the image edge.
![False Negative examples](./figures/FalseNegative.png)

## Part B — Adding a "not sure" option (selective rejection)

Baseline mode always produces an answer. That is useful for statistics,
but for biology it is sometimes safer to defer the most ambiguous
crops.

### How it decides

We split the probability axis into three zones using two thresholds:

- `p(good) ≤ 0.10` → **bad** (accepted as bad)
- `p(good) ≥ 0.75` → **good** (accepted as good)
- `0.10 < p(good) < 0.75` → **rejected / unsure** (deferred, not counted)


### Headline metrics — baseline vs selective (test split)

All selective-rejection metrics below are computed **only on the cells
the model was confident enough to accept.**

| Metric | Baseline | Selective rejection (accepted only) |
|---|---|---|
| Coverage (cells that got a decision) | 100.00% | **98.00%** |
| Accuracy | 98.08% | **98.67%** |
| Recall | 94.71% | **97.27%** |
| Precision | 92.27% | **93.68%** |
| F1 | 93.47% | **95.44%** |
| Specificity | 98.65% | **98.90%** |
| False positives (kept by mistake) | 15 | **12** |
| False negatives (missed good cells) | 10 | **5** |

### Coverage and rejection summary

- Accepted: **1,274 / 1,300** cells (98.00%)
- Rejected / unsure: **26 / 1,300** (2.00%)
  - Among rejected: **6 truly good**, **20 truly bad**

**Intuition.** The model says: "I can give you a decision on **98%**
of cells and be right **~99%** of the time on those; the remaining
**2%** are ambiguous, please look at them yourself." Notably, the
**number of remaining good-cell mistakes drops from 10 to 5** under
selective rejection, while precision goes from 92.3% to 93.7% — so
"kept-as-good" labels are noticeably more trustworthy at almost no
coverage cost.

### Accepted-only confusion matrix — counts

| True \ Pred (accepted) | Pred `bad` | Pred `good` | Row total |
|---|---|---|---|
| True `bad` (accepted)  | **1079** (TN) | **12** (FP) | 1091 |
| True `good` (accepted) | **5** (FN)    | **178** (TP) | 183 |

### Accepted-only confusion matrix — row-normalized (%)

| True \ Pred (accepted) | Pred `bad` | Pred `good` |
|---|---|---|
| True `bad` (accepted)  | 98.90% | 1.10% |
| True `good` (accepted) | 2.73%  | 97.27% |

### Qualitative examples under the selective policy

**Accepted as good, truly good (confident correct accepts):**
![Accepted good examples](./figures/AcceptedGood.png)

**Accepted as bad, truly bad (confident correct rejects):**
![Accepted bad examples](./figures/AcceptedBad.png)

**Rejected / unsure cells:** these are the cells the model refused to
score. This pile is also useful as a QC signal — it highlights cases
where either the human reviewer or the model would benefit from better
image quality / clearer membranes.
![Rejected uncertain examples](./figures/RejectedUncertain.png)

**Accepted but wrong — false positives after filtering (12 cells total):**
![Accepted false-positive examples](./figures/AcceptedFalsePositive.png)

**Accepted but wrong — false negatives after filtering (5 cells total):**
![Accepted false-negative examples](./figures/AcceptedFalseNegative.png)

### What improved

- **Recall** improved (94.7% → **97.3%**) on accepted cells.
- **Precision** improved (92.3% → **93.7%**) on accepted cells.
- **Accuracy** reached **98.7%** on the accepted subset.
- The cost is **only ~2% coverage loss** (~1 cell in 50 deferred for a
  human glance).

## Part C — Biological results on the full dataset

After running the model with the selective-rejection policy on the
full imaging dataset:

- **17,621 candidate cells** were scored across all images.
- Only the cells the model accepted as **good** are taken forward to
  the size analysis below: **2,566 good cells from 277 frogs**.
- Of those, **2,555 cells** also have a matched nucleus, and that
  subset is what drives the nucleus, N/C-ratio, and regression
  analyses.

### C1. Cell-area distribution (good cells only)

![Cell area distribution](./figures/AreaDistribution.png)

- **Median cell area: 293.25 µm²** (IQR 259.88 – 344.85 µm²).
- **Mean ± SD: 308.34 ± 70.23 µm².**
- The distribution is unimodal and slightly right-skewed. We do not see
  two obvious sub-populations.

### C2. Cell-diameter distribution (good cells only)

![Cell diameter distribution](./figures/DiameterDistribution.png)

The long-axis and short-axis diameters are shown side by side. Because
the cells are mostly rounded (median axis ratio ~1.2 — see column
`cell_axis_ratio` in `filtered_areas.csv`), the two distributions
track each other closely.

### C3. Nucleus size distribution

![Nucleus distribution](./figures/NucleusDistribution.png)

### C4. Nucleus-to-cell area ratio (N/C ratio)

![N/C ratio distribution](./figures/NCRatioDistribution.png)

- **Median N/C ratio: 0.136** (nucleus takes up ~14% of cell area).
- **Mean ± SD: 0.141 ± 0.044.**

### C5. Per-frog variation in mean cell area

![Per-frog cell-area boxplot](./figures/PerFrogBoxplot.png)

Per-frog boxplot of cell area (top-40 frogs by cell count, sorted by
per-frog median cell area).

![Per-frog nucleus-area boxplot](./figures/PerFrogNucleusBoxplot.png)

Per-frog boxplot of nucleus area for the same 40 frogs, in the same
order as the cell-area panel above. A frog with large cells but small
nuclei (or vice-versa) will stand out as a mismatch between the two
plots.

### C6. Does the nucleus scale with cell size? (regression)

![Nucleus area vs cell area regression](./figures/NucleusVsCellRegression.png)

We fit a regression in **log-log space** so the slope can be read as a
biological scaling exponent: a slope of **1** means the nucleus grows
in exact proportion to the cell (**isometric**); a slope **below 1**
means the nucleus grows **slower** than the cell (negative allometry);
a slope **above 1** means the nucleus grows **faster**.

Two fits are shown on the same plot:

- **Pooled regression** (black solid line) — uses every cell as if all
  cells were independent. Useful as an overall trend.
- **Mixed-effects regression** (orange dashed line) — first removes
  the overall level differences between frogs, then estimates how
  nucleus and cell move together **within** the same frog. This is
  the more biologically meaningful slope when frogs differ from each
  other.