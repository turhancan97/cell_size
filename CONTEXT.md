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

- Adult segmented data: `/shared/sets/datasets/vision/cellpose/Adults_training`
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

