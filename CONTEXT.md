# Cell Size Repository Context

This file is the handoff document for new AI sessions working in this repository.
Read it before inspecting or changing code. Treat the repository and generated results
as an ongoing scientific analysis project, not as a generic image-classification demo.

## Session Rules

- Preserve existing user changes. Check `git status --short` before editing and do not
  reset, checkout, or overwrite unrelated work.
- Inspect the relevant code, configuration, tests, and generated output before making
  claims about behavior.
- Keep raw outputs available for auditability. Use a new output directory for diagnostic
  reruns instead of overwriting final results.
- When a meaningful decision, experiment, threshold change, result, bug, or test is
  completed, update this file under `Project Log` and refresh `Last updated`.
- Distinguish clearly between measured results, estimates based on classifier
  confidence, and biological interpretations.
- If a result depends on rerunning inference or morphology QC, say so explicitly.

## Project Purpose

This project segments frog red blood cells (RBCs), measures cell and nucleus morphology,
and uses a trained image classifier to select reliable cell measurements. The main
biological measurements include cell area, nucleus area, cell/nucleus area ratio, and
shape/diameter features.

The important experimental design is:

1. Adult frog images are used for segmentation development and classifier training.
2. A cell-quality classifier is trained using adult-frog cell examples.
3. The adult-trained classifier is applied to tadpole images to identify good and bad
   cell crops. This is transfer across biological material, so tadpole domain shift is
   a central diagnostic concern.
4. Good classifier predictions are converted into cell and nucleus measurements.
5. A morphology quality-control (QC) layer removes measurements that are mathematically
   invalid or biologically implausible, especially implausible nucleus/cell ratios.

The pipeline does not currently mean that every cell detected by segmentation becomes a
reported measurement. Segmentation produces candidate masks; classification selects
quality-approved candidates; morphology QC produces the biology-facing clean dataset.

## Pipeline Components

### Segmentation

The `cell-size` command uses Cellpose-SAM-style segmentation configured through Hydra.
The main configurations are:

- `src/cell_size/configs/segmentation/membrane.yaml` for cell/membrane masks.
- `src/cell_size/configs/segmentation/nucleus.yaml` for nucleus masks.
- `cell-size-segment.sh` contains the current cluster example for tadpole data.

Segmentation configuration is data-dependent. In particular, a threshold that works
well in `notebooks/run_Cellpose-SAM.ipynb` may not behave identically in the batch
pipeline if channel selection, preprocessing, image resizing, or model settings differ.
The current checked-in nucleus configuration uses `threshold_value: 0.6`; verify the
file before discussing a new experiment.

### Classifier

The classifier code is primarily in `src/cell_size/classifier/` and the entry point is
`cell-size-classify`. The model checkpoint used in the current workflow is commonly:

```text
./classifier_output/run_2/best_model.pt
```

The standard example is in `cell-size-classify.sh`. Typical inputs are segmented,
per-image folders such as `TIFF_AH_160_05`, containing the source image and masks.

The classifier scans all processed images below `data_dir`; there is no normal CLI
argument that filters by frog ID. To run a diagnostic subset, make a temporary input
directory containing only the selected per-image folders and pass that directory as
`data_dir`. Always use a separate `output_dir` for the diagnostic run.

### Selective Rejection

When enabled, the classifier uses the predicted probability of a cell being good:

- `p_good <= t_bad`: verdict `bad`.
- `p_good >= t_good`: verdict `good`.
- Otherwise: verdict `rejected`.

`filtered_areas.csv` is generated from `good` classifier verdicts only. Therefore:

- Lowering `t_good` can turn previously rejected cells into measured cells.
- Lowering `t_good` does not recover cells already classified as `bad`.
- Lowering `t_bad` from `0.10` to `0.05` changes cells in the interval `(0.05, 0.10]`
  from `bad` to `rejected`; it normally does not increase measured-cell counts.
- A threshold-only simulation using `predictions.csv` can estimate classifier counts,
  but it cannot provide exact new morphology-QC counts unless morphology measurements
  for the newly good cells are regenerated.

The classifier configuration is in `src/cell_size/configs/classifier/default.yaml`.
The current production-style tadpole example has used approximately:

```yaml
classifier:
  confidence_threshold: 0.5
  selective_rejection:
    enabled: true
    t_bad: 0.10
    t_good: 0.76
```

The `confidence_threshold` is ignored when selective rejection is enabled; `t_bad` and
`t_good` control the verdicts.

## Morphology QC

The reusable QC implementation is in:

- `src/cell_size/classifier/morphology_qc.py`
- `src/cell_size/configs/morphology_qc/default.yaml`
- `src/cell_size/qc_main.py`

The standalone entry point is `cell-size-qc-filter`. Classification also runs QC after
writing `filtered_areas.csv` when `morphology_qc.enabled=true`.

The current checked-in QC settings are:

```yaml
morphology_qc:
  enabled: true
  require_nucleus: true
  min_nc_ratio: 0.05
  max_nc_ratio: 0.30
  sensitivity_max_nc_ratios: [0.30, 0.40, 0.50, 0.80]
```

A row is rejected when the nucleus measurement is missing, required numeric values are
missing/non-finite/non-positive, or `nc_ratio` is outside the configured interval.
Each row receives `qc_pass` and `qc_reason` for auditability.

QC output names normally include:

- `filtered_areas_qc.csv`
- `filtered_areas_qc_rejected.csv`
- `frog_aggregated_metrics_qc.csv`
- `morphology_qc_image_summary.csv`
- `morphology_qc_frog_summary.csv`
- `morphology_qc_threshold_sensitivity.csv`
- `frog_aggregated_metrics_qc_comparison.csv`

Raw outputs such as `filtered_areas.csv` and `frog_aggregated_metrics.csv` must remain
unchanged when QC is applied. QC outputs are the preferred biology-facing deliverables;
raw outputs are retained for comparison and sensitivity analysis.

## Frog ID Convention

`src/cell_size/sample_ids.py` is the shared parser. It extracts the alphanumeric token
from names matching:

```text
TIFF_AH_<frog_id>_<image_index>
```

Examples:

```text
TIFF_AH_160_05   -> frog_id "160"
TIFF_AH_001_04   -> frog_id "001"
TIFF_AH_030K_12  -> frog_id "030K"
```

`frog_id` must be treated as a string. Leading zeros and letters are biologically
meaningful. Do not cast it to `int`. Unparseable names are missing and are excluded from
frog-level aggregation with a warning.

## Important Data Locations

These are examples from the current workflow; verify paths on the active machine:

- Adult segmented data: `/shared/sets/datasets/vision/cellpose/Adults/adults_second_trial`
  (the older `.../cellpose/Adults_training` path in earlier notes does not exist)
- Adult training labels: `latest_adults.csv` in the repo root (gitignored), 8,663 hand-labelled cells
- Tadpole segmented data: `/shared/sets/datasets/vision/cellpose/Tadpoles/Tadpoles_training`
- Classifier checkpoint: `./classifier_output/run_2/best_model.pt`
- General classification output root: `./classify_output`
- Adult results: commonly under `classify_output/full_adult_results`
- Final tadpole results: commonly under `classify_output/final_tadpole_results`
- Diagnostic subset results: use a new directory such as
  `classify_output/diagnostic_tgood_0.50`

Do not assume every output directory has the same QC threshold. Read its configuration
or metadata and inspect the files before comparing results.

## Known Scientific Context

### Adult frog 160

Adult frog `160` showed unusually large mean nucleus area and high nucleus/cell ratios.
Inspection indicated segmentation failures in some images, including cases where the
nucleus was not properly separated from the cell. The classifier overlays were useful
for showing that classifier acceptance and segmentation quality are separate issues.
Morphology QC based on nucleus presence, valid measurements, and an upper `nc_ratio`
limit was introduced so frog 160 does not have to be discarded wholesale.

### Tadpole low-cell-count problem

The biology team reported many tadpole individuals with fewer than the target minimum of
40 measured cells. Extreme examples included IDs such as `900K`, `854K`, `883K`, `104K`,
`859K`, and others. The likely causes to investigate are:

- domain shift because the classifier was trained on adult frogs and applied to tadpoles;
- segmentation differences in tadpole images;
- selective-rejection thresholds that are too strict for tadpole appearance;
- morphology QC removing measurements after classification.

The correct diagnosis requires separating these stages. Compare candidate masks,
classifier verdict/confidence distributions, raw good-cell counts, QC-pass counts, and
overlays. Do not conclude that threshold relaxation fixes the problem until the newly
accepted cells have been visually and biologically checked.

## Standard Commands

Install the editable package in the project environment:

```bash
pip install -e .
```

Run segmentation using the relevant data path and configuration, for example:

```bash
cell-size \
  data.data_dir=/path/to/segmented-or-raw-input \
  output.generate_overlays=true \
  output.compute_cell_areas=true \
  output.generate_plots=true
```

Run classification with a new output directory:

```bash
cell-size-classify \
  checkpoint=./classifier_output/run_2/best_model.pt \
  data_dir=/path/to/segmented/data \
  output_dir=./classify_output/diagnostic_run \
  classifier.confidence_threshold=0.5 \
  classifier.selective_rejection.enabled=true \
  classifier.selective_rejection.t_bad=0.10 \
  classifier.selective_rejection.t_good=0.76 \
  morphology_qc.enabled=true \
  morphology_qc.max_nc_ratio=0.30 \
  generate_filtered_overlays=true
```

Generate overlays from an existing predictions file with
`cell-size-classify-overlays` when appropriate. Use `cell-size-qc-filter` when QC must
be rerun on an existing `filtered_areas.csv` without rerunning classifier inference.

## Reporting and Comparison

Biology-facing reports should normally use the QC aggregate, not the raw aggregate.
When comparing two runs, report at least:

- number of frogs and images processed;
- total candidate cells, classifier-good cells, and QC-pass cells;
- number of frogs with QC-pass `n_cells < 40`;
- frogs/images most affected by rejection;
- raw-versus-QC changes in cell area, nucleus area, and `nc_ratio`;
- threshold values and whether values are exact rerun results or estimates.

Useful report/plot code is under `notebooks/`, especially `biology_plots.py` and
filtered-area analysis notebooks. Keep generated reports and spreadsheets separate from
raw pipeline CSVs and record the source path and threshold in any deliverable name.

## Environment Notes

The project requires Python `>=3.10` and uses PyTorch, torchvision, Cellpose-related
dependencies, Hydra, pandas, scikit-image, OpenCV, and tifffile. The working cluster
environment has used CUDA 12.4 PyTorch packages. Previous environment failures included:

- `RuntimeError: operator torchvision::nms does not exist`, usually indicating an
  incompatible torch/torchvision installation;
- `undefined symbol: iJIT_NotifyEvent` while importing torch, caused by an incompatible
  binary/MKL package combination.

When debugging imports, first verify that `python`, `torch`, and `torchvision` come from
the active `cell-size` environment and that their versions are compatible. Do not change
the environment or reinstall large GPU packages without checking the active package
list and preserving the known working combination.

## Project Log

Record new entries at the top in reverse chronological order. Include the date, what
changed, relevant paths/thresholds, verification performed, and unresolved questions.

### 2026-09-15 (later 2) — Newly-accepted cells visually verified; deliverables finalised

- VERIFICATION DONE. 214 overlays rendered and reviewed manually across 7 recovered
  individuals: 104K, 192K, 367K, 465K, 854K, 883K, 900K. Output kept at
  `classify_output/full_tadpole_results/overlay_subset/overlays/`.
- Scope of the check: 10,259 candidate cells shown, 1,223 classifier-good, 751 QC-pass.
  1,060 of the accepted cells are ones the ADULT model had rejected — i.e. 1.6% of all
  65,564 newly-accepted cells, sampled from the individuals that changed most.
- VERDICT: the accepted cells are correct. This closes the "no human has looked at them"
  caveat that had been standing since the run completed. It does NOT make the check a
  census — 800 of 807 individuals remain visually unchecked and rest on the statistical
  case (adult-model recall 48.2% on human labels, held-out F1 0.889, acceptance rate
  17.8% vs human base rate 20.5%).
- Not reviewed, and still worth a look if anyone has time: 100K (the only individual still
  below target, 32 cells) and a high-count control such as 893K for appearance comparison.
- Deliverables finalised:
  - Report artifact updated to v2 (visual check recorded in the verdict box, the decision
    card, the caveat list and the provenance block):
    https://claude.ai/artifact/R9xVjqUPx4y36aprGVebAJ
  - `frog_aggregated_metrics_qc_no_px.xlsx` regenerated for the new run — 807 rows,
    21 columns, `_px` columns dropped, column list verified identical to the July file,
    `frog_id` preserved as text.
- ENVIRONMENT NOTE: `openpyxl` was MISSING from the `cell-size` env, so pandas could not
  write xlsx at all. Installed `openpyxl` 3.1.5 + `et_xmlfile` 2.0.0 with `pip install
  --no-deps` (pure-Python wheels, no compiled parts). Verified afterwards that torch 2.5.1
  and torchvision 0.20.1 still import and are unchanged. Needed again for any future
  spreadsheet deliverable.
- REMAINING OPEN QUESTION for the biology team: `max_nc_ratio=0.30` now discards 8,424
  cells (86% of them in 0.30-0.40); raising to 0.40 returns 7,205. Unchanged by the visual
  check — it is a biological plausibility question, not a classifier question.

### 2026-09-15 (later) — Overlay generation vectorised and parallelised

- `generate_filtered_overlay` had the same O(cells x full-mask) bug as the crop path: it ran
  `masks == label` over the whole 3984x6000 frame once per cell. Replaced with label-indexed
  lookup tables (fill colour, alpha, outline colour, is-good) applied in one vectorised pass;
  `_to_display_rgb` now uses the integer-LUT rescale instead of a float64 copy of the frame.
  Dense image (649 labels): 64 s -> 4 s. Sparse images are unchanged or marginally slower —
  the win scales with cell count, and the mean image has ~39 cells.
- Blend is done one channel at a time in float64, deliberately: float32 was 2x cheaper in
  memory but produced a 1/255 difference. Output is now pixel-identical (max|diff| = 0)
  to the pre-patch implementation, verified on images with 1, 22 and 649 labels.
- `generate_filtered_overlays_from_predictions` parallelised with the same pool pattern as
  step 2 (spawn + `_single_threaded_env()`, `imap_unordered` since each task writes its own
  file), `num_workers` wired through `classify_overlays.yaml` and `classify.yaml`.
- MEASURED: 16 images (900K) in 95 s on 4 workers = 5.9 s/image wall, vs 17.1 s/image serial.
- Overlays still walk the whole `data_dir`, so a review subset needs a directory of symlinks
  to the wanted per-image folders (the diagnostic-subset pattern documented above). The full
  19,445-image set is ~92 h serial and was never regenerated for the tadpole-model run.

### 2026-09-15 — Tadpole-native classifier trained and applied; low-count problem resolved

- HEADLINE: individuals below the 40-cell target went from 92 to 1 (of 807). QC-pass cells
  72,175 -> 121,311. Median cells/individual 87 -> 141. Report artifact:
  https://claude.ai/artifact/R9xVjqUPx4y36aprGVebAJ
- New results: `classify_output/full_tadpole_results/` (SLURM 501013). The July baseline was
  RENAMED `final_tadpole_results` -> `old_tadpole_results`; earlier log entries referring to
  `final_tadpole_results` mean that directory.
- Cause of the change: classifier only. Segmentation was NOT re-run — both runs measure the
  same 765,786 candidate cells from the same masks. Trained efficientnet_b0 (unfrozen) on
  `latest_tadpoles.csv` (7,543 hand-labelled tadpole cells, 1,548 good / 5,995 bad,
  reviewer marcin.czarnoleski@uj.edu.pl, delivered 2026-09-11). Held-out test F1 0.889
  (precision 0.848, recall 0.935); val F1 0.890. Checkpoint:
  `classifier_output/Tadpoles/clf_efficientnet_b0_freezefalse_..._20260914_122813/best_model.pt`.
  Fine-tuning beat frozen-encoder probing by 0.08-0.11 F1 across all three encoders.
- DECISIVE VALIDATION (does not require trusting either model): scored both models against the
  7,541 human-labelled cells. The ADULT model had never seen any of them, so its number is a
  fair estimate — recall on human-"good" cells 48.2%, precision 93.2%. It was not making
  mistakes; it was silently discarding more than half the good cells. That is the low-count
  problem measured at the decision level. New model 96.0% recall on the same set, but it
  trained on ~70% of them — quote 0.889 held-out F1 as the honest figure, never 96.0%.
- Corroboration: human base rate of good cells 20.5%; new model accepts 17.8% of all
  candidates, old accepted 9.9%. 591 of 807 individuals contain NO labelled image at all —
  their median is 146 cells and none is below target. Training contamination is 1.1% of
  measured cells.
- Measurements barely moved, so this is more data of the same kind: medians cell area
  -2.8%, nucleus area +3.4%, nc_ratio +7.0%; per-individual means correlate 0.991 / 0.987 /
  0.968 between runs. The nc_ratio rise (+5.1% on per-individual means) is the one biology
  question worth raising — the new model accepts more large-nucleus cells.
- CAVEAT for any future comparison: this run changed TWO variables — model AND thresholds
  (t_bad 0.10->0.20, t_good 0.76->0.80). Both moved STRICTER, so they cannot explain the
  increase, but it is not a clean single-variable comparison.
- Open items: (1) max_nc_ratio=0.30 now discards 8,424 cells (was 1,308), 86% of them in
  0.30-0.40; raising to 0.40 returns 7,205 — a real decision now, unlike in September when it
  affected a handful. (2) Individual 100K is the only one still short (32 cells, was 7;
  199 candidates, 72 classifier-good — the shortfall is now at QC, not the classifier).
  (3) No human has visually checked any of the 65,564 newly-accepted cells; overlays were not
  generated for this run.

### 2026-09-15 — Pipeline performance work; resume support added

- `classify_main.py` gained `predictions_csv=<path>`: reuses a finished inference pass and
  skips step 1. Added after SLURM 500382 completed 22.8 h of inference and was then killed by
  the 24 h wall 74 min into step 2, which buffers in memory and writes only at the end.
- Step 1 was O(cells x full-mask) — `_crop_cell` ran `np.where(mask == label)` over the whole
  3984x6000 mask once per cell. Added `_label_bboxes()` (one `scipy.ndimage.find_objects`
  pass), an integer-LUT rescale in `_read_image_rgb` (was a float64 copy of the full frame),
  and a single-pass `match_nuclei_to_cells`. Measured 37x / 5.2x / 4.4x on those three;
  read+crop stage 5.9x overall. Verified crops bit-identical on 8 images (9-649 cells).
- Step 2 parallelised over images (`Pool.imap`, spawn, sorted tasks so row order is
  independent of worker count), `num_workers` defaulting to `len(os.sched_getaffinity(0))`.
  MEASURED RESULT: 19,371 images in 15 min on 10 CPUs, vs a 3.2 h serial projection.
- IMPORTANT GOTCHA: the first parallel attempt was 0.9x — SLOWER. BLAS/OpenMP were
  spin-waiting across ~5.6 cores per image while doing less work than one pinned thread
  (0.89 s/img wall at 4.95 s CPU, vs 0.71 s/img at 0.81 s CPU). `_single_threaded_env()` pins
  OMP/MKL/OpenBLAS to 1 thread in the environment the pool spawns from; this is what makes
  process parallelism work at all. Do not remove it.
- Correctness: parallel step 2 output is byte-identical to serial, and both match the July
  reference `filtered_areas.csv` column-for-column on a 120-image random sample and on the
  24 densest images. `scipy` added to `pyproject.toml` (was only transitive via scikit-image).

### 2026-09-09 (later 9) — Adult training cells added as the reference distribution; conclusion sharpened

- Change: Added the actual classifier training data to the report's appearance analysis, so the
  scatter shows where the model's expectation sits rather than only comparing tadpole groups.
  Both artifacts republished at their existing URLs. Report now 5.77 MB.
- DATA LOCATION CORRECTION: the adult root documented in this file as
  `/shared/sets/datasets/vision/cellpose/Adults_training` does NOT exist. The real path is
  `/shared/sets/datasets/vision/cellpose/Adults/adults_second_trial/` (sibling dirs
  `adults_first_trial`, `Adults_training_subset`). The `dataset` column in the label CSV names
  the subdirectory.
- TRAINING LABELS: `latest_adults.csv` in the repo root (gitignored) is the hand-labelled
  training set — 8,663 labelled adult cells, 1,258 good / 7,405 bad, 163 images, 157 frogs,
  dataset `adults_second_trial`, reviewer marcin.czarnoleski@uj.edu.pl, dated 2026-04-28.
  Schema: dataset, image_path, mask_index, verdict, reviewer_email, comment, reviewed_at.
  File mtime 3 May, checkpoint `run_2/best_model.pt` 4 May — consistent. 162 of 163 labelled
  images have image + mask + nucleus_mask on disk.
- Sampled 98 good + 70 bad adult cells from 14 images and measured the same six features
  (`sample_adult.py`, `features_adult.csv`, `features_all2.csv`, `groups_adult.json`).
- KEY RESULT — the control is now inside the training data itself, which is far stronger than
  the previous tadpole-only control. Separation (ROC AUC):
  - adult good vs adult bad: 0.504-0.575 on EVERY feature, i.e. chance. The human labels were
    about broken/overlapping cells, not staining, so these features provably do not measure
    cell quality.
  - adult good vs ext_bad (worst tadpole slides): nucleus darkness 0.979, nucleus contrast
    0.913, cytoplasm saturation 0.820, granularity 0.706, cytoplasm brightness 0.637.
  - adult good vs ok_good (tadpoles above target): nucleus darkness 0.522, contrast 0.608 —
    i.e. the tadpoles that WORK sit on top of the training distribution.
- Nucleus darkness is the cleanest single axis, monotonic away from the training value:
  adult_good 0.357, ok_good 0.357 (identical), aff_bad 0.408, ext_bad 0.533.
- IMPORTANT NUANCE that corrects earlier wording: hue is NOT the discriminator against the
  training set (AUC 0.611). Adult nuclei sit at hue 294, failing tadpoles 288, working tadpoles
  314 — so the adults are closer to the FAILING slides on colour. Earlier drafts said the
  rejected cells are "paler, bluer and more granular than the training material"; the "bluer"
  half is wrong against the training set, though still true against working tadpole slides
  (AUC 0.893). The report now leads with paleness and loss of nucleus/cytoplasm contrast, and
  states explicitly that colour is not the explanation. Findings card 3 reworded accordingly.
- So the defensible claim is narrower and better: the model tolerates real shift (working
  tadpoles differ from adults in hue and cytoplasm saturation and still classify fine); what it
  does not tolerate is the nucleus being pale and poorly separated from its own cytoplasm.
- Page changes: new `--ref`/`--ref-2` green tokens (theme-aware) for the training groups; the
  two adult groups added to the section 03 gallery selector with the left side now defaulting to
  "Adult training · marked GOOD"; scatter defaults to adult_good + ok_good + ext_bad; feature
  table restructured to five median columns (adult good, adult bad, tadpole above, below, far
  below) with the control column now "within training".
- Speech script section 04 rewritten around the training anchor; script is now ~2,057 spoken
  words, about 16 minutes, and the stated timings in the header and opening were updated to match.
- Verification: HTML tag balance checked with a parser over the body (no mismatches, nothing
  unclosed) after an earlier draft of the patch introduced a stray `</p>` inside a note div,
  which was caught and fixed before writing; `node --check` on both pages' JS; every
  substitution asserted on an exact match count (20 in the report patch). Layout not visually
  verified — no headless browser here.

### 2026-09-09 (later 8) — max_nc_ratio sensitivity measured at production settings; report figure corrected

- Question answered: does raising `morphology_qc.max_nc_ratio` above 0.30 recover meaningful
  numbers of cells? Measured across all 807 individuals with the classifier UNCHANGED at
  `t_good=0.76`, by recomputing the row-level QC predicate from
  `filtered_areas_qc.csv` + `filtered_areas_qc_rejected.csv` (mirrors `apply_morphology_qc`:
  valid area_px, valid nucleus_area_px, valid nc_ratio, nc_ratio >= 0.05).
- Answer: a real but small and highly concentrated effect that SATURATES by 0.35-0.40.
  QC-pass cells / individuals below 40:
    0.30 (current) 72,175 / 92     0.35 73,102 (+927) / 85
    0.40 73,273 (+1,098) / 85      0.50 73,333 (+1,158) / 84
    0.80 73,372 (+1,197) / 84      1.00 73,461 (+1,286) / 83
  Removing the limit entirely only ever rescues 9 of the 92, so nothing useful lies above 0.40.
- Seven individuals cross 40 at max_nc_ratio=0.40: 049K 32->43, 073K 39->40, 097K 39->44,
  192K 35->53, 418K 30->45, 426K 38->44, 465K 39->51.
- Where the +1,098 recovered cells land: 72.6% go to individuals ALREADY above 40, 15.8% to
  the below-target band, 0.5% to far below. Median gain per individual is 0. The 12 extreme
  individuals gain 4 cells between them, all of it in 104K (5->9); 900K, 854K, 883K, 859K,
  870K, 856K, 865K, 855K, 877K, 860K, 852K all gain zero. Reason: `max_nc_ratio` filters
  cells the classifier already accepted, and 900K has only 4 accepted cells in total — no QC
  setting can create cells that never passed the classifier.
- ERROR CORRECTED in both deliverables: the report's section 08 decision card claimed "367K
  would go from 23 to 55 cells at 40%". That conflated two changes — 55 required `t_good`
  lowered to 0.50 AS WELL. At production `t_good=0.76`, relaxing the nucleus limit alone takes
  367K from 23 to 39, still short of 40. Both the report card and the speech script now quote
  the correct production-setting figures (7 of 92 cross, 192K 35->53, extreme cases gain 4
  cells between them). LESSON: never quote a number from the diagnostic run as if it came from
  a single-parameter change; `diagnostic_tgood_0.50` differs from `final_tadpole_results` in
  BOTH t_good and, in the recomputes, max_nc_ratio.
- Standing conclusion for the meeting: relaxing max_nc_ratio to 0.40 is worth doing ONLY if a
  nucleus above 30% of cell area is biologically plausible for tadpole RBCs. It is a separate,
  minor improvement for borderline individuals in the mid-30s — not a fix for the low-count
  problem, and no help at all for the extreme cases.

### 2026-09-09 (later 7) — Speaking notes prepared as a second artifact

- Change: Wrote a plain-language speaking script for the biology-team meeting, published as a
  separate companion artifact (the diagnostic report is unchanged).
  Script: https://claude.ai/code/artifact/0cfc1dfc-5284-426a-83bb-4e7295223f64
  Report: https://claude.ai/code/artifact/f43894a7-1c14-4a93-815d-4e237e648c6e
- Structure: 11 blocks — opening, one per report section 01-09, closing summary — plus an
  "If someone asks" section with six anticipated questions. 1,831 spoken words, about 14
  minutes at 130 wpm; per-section time estimates shown.
- Written for a non-native, non-computational audience: short sentences, no jargon (no
  "domain shift", "distribution", "ROC AUC"), and numbers spoken as ratios — "93 cells out
  of every 100" rather than "93.06%". Presenter cues are visually distinct (small, uppercase,
  grey, chevron-marked) so they are not read aloud by mistake.
- The script keeps the honest caveats deliberately: the two sparse slides (104K img01 has 3
  detected cells, 367K img14 has 9), that 893K in the 8xxK series worked fine, that
  retraining means re-running all tadpoles so counts shift slightly everywhere, and that
  whether the cells are biologically normal is the team's call and not established by us.
- Source file `speech.html` in the session scratchpad. Uses localStorage only, to dim
  sections already covered; no capabilities declared.
- Verification: spoken-word count measured from `<p>` inside `.script` blocks only (cues
  excluded) and the stated timing corrected from 13 to 14 minutes to match; `node --check`
  on page JS. Layout not visually verified (no headless browser).

### 2026-09-09 (later 6) — Section 07 rewritten as a plain-language threshold explainer

- Change: Section 07 now explains what `t_bad` and `t_good` actually do, in language for a
  biologist, and shows why moving them cannot recover the missing cells. Same URL, 5.2 MB.
  Report prose is now ~1357 words (up from 707 — this section deliberately carries more).
- New content: three cards defining the settings (t_bad 0.10 discard / 0.10-0.76 not sure /
  t_good 0.76 measure), then a hand-authored inline-SVG figure, then the existing rerun
  table unchanged.
- The figure: the 0-1 cell-score axis cut by the two thresholds into three zones, with the
  share of all detected cells in each zone, one row for the 92 below target and one for the
  715 above. Zone widths follow the score axis, so the leftmost zone spans only 0.00-0.10
  yet holds most cells — that mismatch is the whole point of the picture.
- MEASURED DISTRIBUTION (new numbers, from `predictions.csv`, confidence = p_good):
  - Below target (75,996 cells): under 0.10 = 93.06%, 0.10-0.76 = 3.51%, >=0.76 = 3.43%.
    90.66% of all cells sit in 0.00-0.02 alone.
  - Above target (689,790 cells): under 0.10 = 86.00%, 0.10-0.76 = 3.33%, >=0.76 = 10.67%.
    9.54% sit in 0.90-1.00, against 2.59% below target.
  - So the scores are bimodal, and the two cohorts differ almost entirely at the CONFIDENT
    ACCEPT end (10.67% vs 3.43%), not in the borderline middle (3.33% vs 3.51%). What went
    missing is confident acceptances. The middle band is the ceiling on anything `t_good`
    can recover: 3.5% of cells.
- Precision point worth keeping: `t_bad` is the weaker lever. Lowering it only moves cells
  from "discarded" to "not sure" — it does not add measured cells, because `t_good` alone
  decides what gets measured. The report now says this explicitly, because the user has
  been running `t_bad=0.05` in `cell-size-classify.sh`. An earlier draft claimed "no setting
  reaches them", which was wrong, and was corrected before publishing.
- Note on the working tree: `cell-size-classify.sh` currently holds the DIAGNOSTIC config
  (`t_bad=0.05`, `t_good=0.50`, `data_dir=.../tadpole_diagnostic_subset`,
  `output_dir=./classify_output/diagnostic_tgood_0.50_tbad_0.05_`), not the production
  values. The report's stated values (0.10 / 0.76) are those that produced
  `final_tadpole_results`. Do not read the script as the production config.
- Verification: SVG zone boundaries recomputed from the axis mapping and confirmed to match
  the drawn rect x/width values exactly (112 / 174.4 / 586.2 / 736 for scores 0 / 0.10 /
  0.76 / 1.00); all text y-coordinates confirmed inside the viewBox; zone percentages sum to
  100.0 in both rows; `node --check` on page JS. Layout not visually verified.

### 2026-09-09 (later 5) — Report prose trimmed for live presentation

- Change: Cut the report's text roughly in half so it can be talked through in a meeting
  rather than read. Every table, chart, image, the band key and the decision cards are
  unchanged — text only. Same URL, 5.19 MB.
- Prose went from ~2000 to ~707 words across the nine sections. Per section now: 01 52 /
  02 42 / 03 111 / 04 133 / 05 118 / 06 67 / 07 77 / 08 99 / 09 8 words.
- Approach: kept every number and claim, dropped the connective explanation around them.
  Removed prose that duplicated a figure's own labels — the scatter's axis titles already
  name both axes, the whole-slide colour key already explains the outline colours, and the
  feature table's caption already defines the separation scale, so the paragraphs restating
  those went. Figure captions were tightened but not gutted, since they are needed to read
  the figures unaided.
- Reason (from the user): they will present this live and write their own speaking notes per
  section, so the page should carry the evidence and the headline claims, not the argument in
  full sentences. Keep this in mind for any future edits — do not re-expand the prose.
- Verification: all 31 + 11 substitutions asserted on exact match counts; programmatic check
  that all 13 figure/table/interactive anchors survive (`nimgTable`, `funnel`, `cmp`,
  `specTabs`, `scatter`, `featTable`, `slides`, `tiers`, `serChart`, `thrTable`, `roster`,
  `bands`, `decisions`); `node --check` on page JS. Pre-trim copy kept as `report4.bak` in
  the scratchpad. Layout not visually verified (no headless browser).

### 2026-09-09 (later 4) — Live labelling exercise removed from the report

- Change: Removed the "Would you measure these?" section at the user's request. The report
  is now nine sections and purely read-only. Same URL.
- Removed: the section markup, its CSS, its JS, the `window.claude.use("db")` call, and the
  `db` capability declaration (republished with `capabilities: {}`). Verified the `labels`
  collection was empty first, so no biology-team verdicts were lost. Section numbering
  re-derived positionally; residual-string grep confirms zero references remain.
- Consequence to remember: the report no longer collects labels, so the tadpole fine-tuning
  set has to be produced another way. The section 08 recommendation card now says a set of
  rejected cell images will be sent for labelling separately, and the footer no longer
  claims the report tests the staining hypothesis. The crop-sampling script idea (extract
  rejected crops to a folder plus a spreadsheet for verdicts) is the fallback route and has
  not been built.
- Current report sections: 01 summary table / 02 where cells are lost / 03 what rejected
  cells look like / 04 difference is measurable / 05 whole-slide gallery (24 slides) /
  06 series clustering / 07 why thresholds do not fix it / 08 what we propose /
  09 all 92 individuals.
- Files: scratchpad `report4.html`, `report_final.html` (5.2 MB).

### 2026-09-09 (later 3) — Whole-slide section expanded to 24 slides; density claim corrected

- Change: Added the five slides the biology team asked for plus a matched well-above-target
  and just-above-target example for each, so section 05 now shows 24 slides in three
  columns of 8 (plus the 176K/854K headline pair). Same URL; page now 5.2 MB.
- Requested slides and their real bands (they span TWO bands, not one):
  - far below target: 104K img01 (individual total 5), 883K img08 (4), 900K img13 (3)
  - below target: 192K img02 (35), 367K img14 (23)
  Third column is therefore headed "Short of target" and every card carries a band chip
  (well above / just above / below target / far below) so the distinction stays visible.
- Matched examples chosen by accepted-cell count, NOT by density. Sorting candidate images
  by cell count surfaces dense clumped fields where the classifier rejects almost
  everything (e.g. 207K img01: 321 detected, 2 accepted) - those look like failures and
  would misrepresent a working individual. Added: well above - 326K, 992K, 312K, 280K,
  991K; just above - 155K, 594K, 272K, 409K, 342K.
- CORRECTION to the previous entry: the claim "the failing slides are the densest" is too
  strong and the report now says so explicitly. Two of the requested slides are genuinely
  sparse - 104K img01 has only 3 detected cells, 367K img14 has 9 - so on those a thin
  smear is a real contributing factor alongside classifier rejection. The dense-and-rejected
  pattern still holds for most (447K img13 149 detected / 0 measured, 852K img09 146/1,
  192K img02 84/1), and the cohort averages are 32.9 detected per image below target vs
  40.3 above, but low counts are NOT always a single cause. Do not present the sparse cases
  as rejection failures.
- Files: scratchpad `overlays_new2.json`, `slide_meta.json` (now carries a `band` field per
  slide), `report4.html`, `report_final.html`.
- Verification: asserted match counts on all 7 substitutions; programmatically confirmed all
  24 tier slides have both an overlay image and metadata, and printed the band membership of
  each column; `node --check` on page JS. Layout not visually verified (no headless browser).

### 2026-09-09 (later 2) — Report labels unified to target-relative bands

- Change: Fixed a real inconsistency in the meeting report. The word "moderate" carried two
  contradictory meanings across sections, and one individual (422K) was labelled both ways.
  All labels are now stated relative to the 40-cell target, and a four-band key is defined
  once in section 03 and reused everywhere. Same URL.
- The bug (worth remembering as a reporting-hygiene lesson): section 04's cell groups used
  `aff_*` = "moderately affected", which were individuals BELOW target (11-38 cells);
  section 05's slide tiers used "partly affected" for individuals ABOVE target (52-68
  cells). 422K (68 cells) therefore appeared as part of the working reference group in
  section 04 and as "partly affected" in section 05. Cause: section 05's tiers were chosen
  to span the outcome range for visual contrast, without checking them against labels
  already used two sections earlier.
- Canonical bands now used throughout (by measured cells per individual, target = 40):
  - Well above target: 150+ (section 05 working column: 125K 159, 893K 193, 534K 166)
  - Above target: 40-149 (section 03/04 reference group `ok_*`, sampled 68-159)
  - Below target: 15-39 (section 03/04 `aff_*`, sampled 11-38 - note the sample dips to 11,
    so the page states the sampled range rather than implying a clean 15-39 cut)
  - Far below target: under 15 (section 03/04 `ext_*` sampled 3-13; section 05 failing
    column 447K 14, 852K 13, 856K 6)
- Group labels are now "Accepted/Rejected · above target | below target | far below target",
  each carrying its sampled cell range in the UI so a reader never has to infer the band.
  "the 92 affected individuals" is now "the 92 individuals below target" throughout.
- Files: scratchpad `patch3.py`, `report4.html`, `report_final.html` (3.66 MB).
- Verification: every substitution asserted on an exact expected match count (23 + 8, all
  matched); grep confirms zero remaining occurrences of the old labels; `node --check` on
  page JS. Layout still not visually verified - no headless browser in this environment.

### 2026-09-09 (later) — Whole-slide section expanded to three outcome tiers

- Change: Section 05 of the meeting report now shows 11 whole-slide overlays instead of 2 —
  the original 176K/854K pair as the headline comparison, plus a three-column tier gallery.
  Same URL. Page is now 3.65 MB.
- Tiers (one column each, 3 slides per column), with per-image verdict counts and the
  individual's overall total in each caption:
  - Working normally: 125K img05 (21 measured / 46 detected, 159 total), 893K img25
    (19/52, 193), 534K img09 (19/81, 166)
  - Partly affected: 422K img06 (21/67, 68), 178K img01 (18/40, 52), 340K img05 (16/38, 63)
  - Failing: 447K img13 (0/149, 14), 852K img09 (1/146, 13), 856K img27 (0/141, 6)
- Finding worth carrying forward: 447K image 13 has 149 detected cells and ZERO measured;
  852K img09 has 146 detected and 1 measured. The failing slides are not sparse — they are
  among the densest in the sample. That kills any residual "not enough cells on the slide"
  reading of the low counts.
- Nuance added honestly to the report: 893K is in the 8xxK series but works fine (193 cells
  measured). So the batch effect is a subset of slides WITHIN the affected series, not the
  whole series. Do not overstate the series story: 8xxK is 65% affected, not 100%.
- Files: scratchpad `patch2.py`, `overlays_new.json`, `slide_meta.json`, `report3.html`,
  `report_final.html`. Overlays downscaled to 1400 px wide at JPEG q72 (~100-210 KB each).
- Verification: `node --check` on page JS; section numbering re-derived positionally (the
  sequential string-replace approach corrupts numbering — always renumber by position).
  Still no headless browser available, so layout remains visually unverified.

### 2026-09-09 — Expanded cell-appearance evidence; domain shift quantified

- Change: Expanded the meeting report's evidence section (same URL,
  https://claude.ai/code/artifact/f43894a7-1c14-4a93-815d-4e237e648c6e) from ~60 cell
  crops to 498, and added a new section 04 quantifying the appearance difference. Sections
  renumbered to 01-10.
- Sampling design (the point is the controls): four groups drawn so that accepted and
  rejected cells come from the SAME slides within each cohort, which controls staining
  within each comparison.
  - `ok_good` 64 / `ok_bad` 48 - accepted / rejected on well-performing slides, 8 frogs
    (345K, 197K, 324K, 260K, 335K, 125K, 422K, 334K)
  - `aff_bad` 120 / `aff_good` 36 - moderately affected, 12 frogs
  - `ext_bad` 200 / `ext_good` 30 - worst-affected, 12 frogs (854K, 900K, 856K, 870K,
    865K, 877K, 883K, 104K, 852K, 855K, 859K, 860K)
- Measured appearance features (`features_all.csv`, 469 cells with segmented nuclei):
  nucleus/cytoplasm contrast, cytoplasm stain saturation, nucleus hue, nucleus darkness,
  nucleus granularity, cytoplasm brightness. Computed from cell mask + nucleus mask at the
  fixed 10-bit scale.
- KEY RESULT (separation vs `ok_good`, ROC AUC):
  - `ok_bad` (control, same slides): 0.51-0.66 on every feature. These features do NOT
    track cell quality.
  - `ext_bad`: nucleus darkness 0.972, cytoplasm saturation 0.953, nucleus contrast 0.941,
    nucleus hue 0.893, granularity 0.714.
  - `aff_bad`: intermediate, 0.62-0.81 - the effect is a gradient, not binary.
  That contrast (control ~0.5 vs worst ~0.95 on the same features) is the actual evidence
  for covariate shift: the features discriminate WHICH SLIDE a cell came from, not whether
  it is measurable. Use this framing, not "the nuclei look odd".
- Exposure confound ruled out: cytoplasm brightness is close across groups (ok_good 0.935,
  ext_bad 0.886), and nucleus contrast - a within-cell ratio, invariant to overall
  exposure - still separates at AUC 0.941. So the affected slides are not merely brighter;
  the nuclei are genuinely less distinct from their own cytoplasm.
- `ext_good` (the rare accepted cells on affected slides) cluster with `ext_bad`, not with
  `ok_good` (nucleus darkness 0.496 vs 0.533 vs 0.357). Even the cells we did measure on
  those slides are off-distribution.
- Files: scratchpad `sample.py`, `sample2.py`, `groups.json`, `groups_ext.json`,
  `features_all.csv`, `dist.json`, `patch.py`, `report_final.html`. NOTE the scratchpad is
  session-scoped and was already cleared once mid-project - regenerate from the scripts, or
  recover the published page with the Artifact tool's `read` action, which is how this
  revision was built after the earlier files were lost.
- Verification: `node --check` on the page JS; section numbering verified positionally
  after an earlier sequential-replace bug renumbered sections wrongly. No headless browser
  in this environment, so the rendered layout was again not visually inspected.
- Remaining questions: unchanged. The biology team still has to say whether the granular
  pale-nucleus cells are healthy tadpole erythrocytes (retrain) or degraded material
  (re-image), and whether 8xxK/0xxK were prepared differently.

### 2026-09-08 (later 3) — Meeting report published; staining hypothesis sharpened

- Change: Built an interactive diagnostic report for the biology-team meeting, published as an
  Artifact: https://claude.ai/code/artifact/f43894a7-1c14-4a93-815d-4e237e648c6e
  Source and embedded assets are in the session scratchpad (`report_final.html`,
  `assets.json`, `report_data.json`); regenerate rather than treating those as durable.
- Files/outputs: report sections cover the `n_images` misreading, the three-stage funnel,
  cell-level image evidence, whole-slide overlays, the series batch effect, the t_good
  rerun, proposed next steps, and a searchable table of all 92 individuals.
- New evidence (this refines the previous entry's hypothesis):
  - Cut individual cell crops straight from the source TIFFs with masks, at a FIXED
    intensity scale (data is 10-bit, 0-1023; scale = value/1023*255) so staining is
    comparable between individuals. Per-crop min/max normalisation must NOT be used here —
    it erases exactly the difference under investigation.
  - The crops show the mechanism far more clearly than the classifier overlays do, because
    the overlay JPGs are colour-tinted by the mask shading. Accepted cells in 176K have
    pink cytoplasm and a compact, dark, smooth nucleus. Rejected cells in 854K and 900K are
    intact, well-formed, single cells whose nuclei are BLUE-VIOLET and visibly GRANULAR.
    This looks like a stain difference (different batch/timing/protocol), not a quality
    difference. Earlier entry framed this as "granular mottled nucleus"; the crops show the
    colour shift is as diagnostic as the texture.
  - 176K's own rejected cells are genuinely poor (damaged, overlapping), so the classifier
    is not broken in general — it fails specifically on this appearance. That control is
    included in the report as a tab.
- Live labelling exercise: the report declares the `db` capability and stores biology-team
  verdicts as `labels/<image>__<mask_index>` documents with `{verdict: "measure"|"skip",
  at: <epoch ms>}`. 24 rejected cells from 854K/900K/856K are presented. Read them back with
  the Artifact tool, `action: "read_db"`, `db_op: "list"`, `collection: "labels"`. These
  verdicts are the seed of the tadpole fine-tuning set.
- Verification: all report figures come from `final_tadpole_results` and the
  `diagnostic_tgood_0.50` rerun; the 92-row list reproduces the biology team's table exactly.
  Page JS syntax-checked with `node --check`. No headless browser was available in this
  environment, so the rendered layout was not visually inspected before publishing.
- Remaining questions: unchanged from the previous entry, plus — collect the labelling
  results after the meeting and use them to decide retrain vs re-image.

### 2026-09-08 (later 2) — Biology-team low-count report diagnosed as classifier domain shift

- Change: Reviewed Marcin's post-summer message listing 92 tadpole individuals below the
  40-cell target. Verified his table against our outputs and traced the loss through the
  segmentation -> classifier -> QC funnel for the whole 807-frog tadpole cohort.
- Files/outputs: `classify_output/final_tadpole_results/{predictions.csv,
  frog_aggregated_metrics_qc.csv,morphology_qc_frog_summary.csv}`; overlays
  `TIFF_AH_854K_01_filtered_overlay.jpg` (0 good / 48 bad) and
  `TIFF_AH_176K_05_filtered_overlay.jpg` (17 good / 16 bad / 3 rejected).
- Verification:
  - His 92 rows reproduce `frog_aggregated_metrics_qc.csv` exactly (`n_images` and
    `n_cells` match on every row, zero mismatches). He is reading the QC table, which is
    the correct biology-facing file.
  - IMPORTANT REPORTING BUG (interpretation, not code): `n_images` in
    `frog_aggregated_metrics_qc.csv` is `nunique(image_path)` computed over QC-passing
    rows only, i.e. the number of images that yielded at least one measured cell — NOT
    the number of images acquired or processed. Examples: 900K reports 3 images but 16
    were segmented and classified; 104K reports 3 but 39 were processed; 854K reports 4
    but 20; 856K reports 5 but 28. The biology team is likely reading these as "we only
    photographed 3 slides", which understates the available data by 4-13x. Consider
    emitting `n_images_processed` alongside `n_images_with_cells`.
  - Funnel, 92 low frogs vs 715 others: candidates/image 32.9 vs 40.3 (segmentation is
    NOT the bottleneck); classifier `good` rate 3.43% vs 10.67%; QC retention of good
    cells 84.2% vs 95.1%. Final yield 0.95 vs 4.08 measured cells per image. The loss is
    overwhelmingly at the classifier stage.
  - The rejection is confident, not borderline: among cells verdicted `bad` in the low
    frogs, median `p_good` = 0.0 and only 1.08% exceed `p_good` 0.05. There is no large
    pool of near-threshold cells to recover, which is why the `t_good` 0.76 -> 0.50
    diagnostic gained so little (see previous entry).
  - Visual confirmation: `TIFF_AH_854K_01` contains 48 cleanly segmented, well-separated
    cells with visible nuclei and the classifier called every one `bad`. Segmentation
    quality is good; the classifier is wrong.
  - Probable mechanism (visual, needs confirmation): in the working frog 176K the
    accepted cells are pale with a small compact dark nucleus, while the cells 176K's
    classifier rejects have a dark-red, granular, mottled nucleus. Every cell in 854K has
    that granular-nucleus/eosinophilic appearance. The adult-trained classifier appears to
    have learned "granular mottled nucleus = bad" as a quality cue, and a subset of
    tadpole slides trips it on every cell.
  - Batch effect: failures cluster by ID series. 8xxK: 65.3% of 49 frogs below 40,
    median good rate 4.5%. 0xxK: 37.1% of 70 below 40. All other series are 0-14%
    (2xxK/3xxK/9xxK/6xxK are ~1-3%). Roughly 58 of the 92 low frogs come from the 8xxK
    and 0xxK series alone. This is consistent with a staining/acquisition batch, not with
    biology.
- Decision/recommendation: threshold and QC relaxation cannot fix this and would only
  trade a count increase for unvalidated cells. The fix is to fine-tune the classifier on
  tadpole crops, with labelling effort concentrated on the 8xxK and 0xxK series. Adult
  results are unaffected and should not be rerun.
- Remaining questions:
  - Confirm the granular-nucleus hypothesis by sampling `bad` crops from 8xxK/0xxK and
    having the biology team label them good/bad; that labelled set doubles as fine-tuning
    data.
  - Are these cells biologically normal for tadpoles, or genuinely degraded slides? Only
    the biology team can answer, and the answer decides whether we retrain or re-image.
  - Whether `max_nc_ratio=0.30` (adult-derived) is right for tadpoles remains open from
    the previous entry.

### 2026-09-08 (later) — t_good 0.50 diagnostic evaluated

- Change: Evaluated the completed diagnostic run `classify_output/diagnostic_tgood_0.50`
  against the `t_good=0.76` baseline `classify_output/final_tadpole_results`. This
  answers the open question logged earlier the same day with exact rerun numbers, not
  a threshold-only simulation: the diagnostic directory contains its own
  `filtered_areas_qc.csv`, so morphology measurements for newly accepted cells exist.
- Files/outputs:
  - `classify_output/diagnostic_tgood_0.50/{predictions,filtered_areas,filtered_areas_qc,filtered_areas_qc_rejected}.csv`
  - `classify_output/diagnostic_tgood_0.50/morphology_qc_{frog_summary,threshold_sensitivity}.csv`
  - `classify_output/diagnostic_tgood_0.50/overlays/` (not yet visually reviewed)
  - baseline: `classify_output/final_tadpole_results/` (same `max_nc_ratio=0.30`, so QC
    counts are directly comparable)
- Thresholds or dataset: 7-frog tadpole subset `104K, 192K, 367K, 465K, 854K, 883K, 900K`;
  identical 10259 candidate masks in both runs; `t_bad=0.10`; `t_good` 0.76 vs 0.50;
  `min_nc_ratio=0.05`, `max_nc_ratio=0.30`.
- Verification (measured, both runs are real reruns):
  - Classifier-good cells 179 -> 239 (+60). Cells verdicted `bad` are unchanged
    (10020 in both runs), as expected: lowering `t_good` only converts `rejected` cells.
  - QC-pass cells 113 -> 148 (+35).
  - Newly accepted cells are morphologically ordinary, not junk: QC pass rate 58.3%
    (35/60) versus 63.1% (113/179) for previously accepted cells; among QC-pass cells
    `nc_ratio` mean 0.238 (new) vs 0.255 (old), `area_um2` 267 vs 291,
    `nucleus_area_um2` 63.5 vs 73.5. QC rejections of new cells are almost entirely
    `nc_ratio_above_max` (23 of 25).
  - Per-frog QC-pass, 0.76 -> 0.50: 104K 5->7, 192K 35->42, 367K 23->33, 465K 39->46,
    854K 4->6, 883K 4->6, 900K 3->8. Only 192K and 465K cross the 40-cell target.
  - Recomputing QC from the diagnostic `nc_ratio` column at looser upper limits
    (classifier `t_good=0.50` held fixed): `max_nc_ratio=0.40` gives 104K 11, 192K 65,
    367K 55, 465K 62, 854K 6, 883K 6, 900K 8 (3 of 7 frogs >= 40); `0.50` adds almost
    nothing beyond that (3 of 7). Subset totals from
    `morphology_qc_threshold_sensitivity.csv`: 148 / 213 / 223 / 228 QC-pass at
    `max_nc_ratio` 0.30 / 0.40 / 0.50 / 0.80 of 239 classifier-good cells.
- Findings:
  1. Lowering `t_good` from 0.76 to 0.50 does materially increase valid cells
     (+31% QC-pass on this subset) and does not visibly degrade the accepted
     population's morphology, but it is not sufficient to fix the low-count problem.
  2. For 192K, 367K and 465K the binding constraint is morphology QC, not the
     classifier: `max_nc_ratio=0.30` is what holds 367K below 40.
  3. For 104K, 854K, 883K and 900K neither relaxation helps. These frogs have
     271 / 533 / 148 / 512 candidate masks of which ~97-99% are verdicted `bad` even at
     `t_good=0.50`. Their bottleneck is upstream — segmentation quality or classifier
     domain shift on tadpole material — and cannot be reached by threshold tuning.
- Remaining questions:
  - Are the newly accepted cells correct on inspection? The overlays in
    `classify_output/diagnostic_tgood_0.50/overlays/` have not been visually or
    biologically reviewed. Do not adopt `t_good=0.50` for production before that.
  - Is `max_nc_ratio=0.30` biologically right for tadpoles? Tadpole RBC nuclei may
    legitimately occupy a larger cell fraction than adult RBCs, in which case 0.30 is an
    adult-derived limit being misapplied. This needs a biological decision, not a
    count-maximising one.
  - For the ~99%-bad frogs, inspect candidate masks and classifier confidence
    distributions directly to separate segmentation failure from domain shift.

### 2026-09-08

- Added this handoff document as the persistent AI-session context.
- Current repository work includes generic string `frog_id` extraction, morphology QC,
  classifier overlay support, and analysis/reporting changes. The worktree is already
  modified; inspect `git status` and diffs before editing.
- Current open scientific question: whether lowering tadpole classifier thresholds,
  especially `t_good` from `0.76` to `0.50`, materially increases valid QC-pass cells.
  Exact QC counts require rerunning morphology measurements for newly accepted cells.

### Template for future entries

```text
### YYYY-MM-DD

- Change:
- Files/outputs:
- Thresholds or dataset:
- Verification:
- Remaining question:
```

