# Cell Size Estimator

Batch cell segmentation and size estimation for microscopy images using
[Cellpose-SAM](https://github.com/MouseLand/cellpose), driven by
[Hydra](https://hydra.cc/) configuration.

## Features

- **Batch processing** -- point at a directory and segment every image in one
  command.
- **Membrane or nucleus** segmentation via preset configs.
- **Hydra configuration** -- override any parameter from the CLI; supports
  multirun for parameter sweeps.
- **Resume support** -- re-run safely; already-processed images are skipped.
- **Auto pixel-scale detection** from OME-TIFF metadata, with manual fallback.
- **Multiple output formats** -- masks as 16-bit TIFF or NumPy `.npy`.
- **Optional overlays and histograms** for quality checking.
- **Catalog CSV** recording every processed image with metadata.

## Installation

### Option 1: conda (recommended)

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd cell_size

# Create and activate a new conda environment
conda create -n cell-size python=3.10 -y
conda activate cell-size

# Install PyTorch with CUDA (adjust cuda version to match your driver)
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install the package in editable mode
pip install -e .

# Install the cellpose submodule
pip install -e cellpose/
```

### Option 2: pip + venv

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd cell_size

# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in editable mode
pip install -e .

# The cellpose submodule is imported at runtime via sys.path.
# If you prefer, you can also install it:
pip install -e cellpose/
```

> **GPU**: Make sure PyTorch with CUDA is installed for GPU acceleration.
> See <https://pytorch.org/get-started/locally/>.

## Quick Start

Segment all `.tif` images in a directory (membrane mode, default):

```bash
cell-size data.data_dir=/path/to/images
```

Segment nuclei instead:

```bash
cell-size data.data_dir=/path/to/images segmentation=nucleus
```

Use multiple file types with resize and overlays:

```bash
cell-size \
  data.data_dir=/path/to/images \
  data.file_types='[".tif",".jpg",".png"]' \
  segmentation.resize=1000 \
  output.generate_overlays=true \
  output.compute_cell_areas=true \
  cell_type=FrogBlood
```

Force reprocessing of all images:

```bash
cell-size data.data_dir=/path/to/images force=true
```

Hydra multirun (parameter sweep):

```bash
cell-size -m \
  data.data_dir=/path/to/images \
  segmentation.flow_threshold=0.3,0.4,0.5
```

## Configuration

All configuration lives in `src/cell_size/configs/` and follows Hydra conventions.

### Main config (`src/cell_size/configs/config.yaml`)

| Key         | Default     | Description                        |
|-------------|-------------|------------------------------------|
| `cell_type` | `"Unknown"` | Label for the Cell_Type CSV column |
| `force`     | `false`     | Reprocess already-segmented images |

### Data (`src/cell_size/configs/data/default.yaml`)

| Key            | Default    | Description                                            |
|----------------|------------|--------------------------------------------------------|
| `data_dir`     | (required) | Path to the image dataset                              |
| `file_types`   | `[".tif"]` | Image extensions to look for                           |
| `recursive`    | `true`     | Scan subdirectories                                    |
| `channels`     | `null`     | Channel indices to select (null = use all)             |
| `pixel_to_um`  | `null`     | Manual µm/pixel value (null = auto-detect from metadata) |

### Segmentation (`src/cell_size/configs/segmentation/`)

Two presets: `membrane` (default) and `nucleus`.

| Key                   | membrane | nucleus | Description                           |
|-----------------------|----------|---------|---------------------------------------|
| `target`              | membrane | nucleus | Descriptive label                     |
| `chan`                 | 0        | 1       | Primary channel for cellpose          |
| `chan2`                | 0        | 0       | Secondary channel                     |
| `flow_threshold`      | 0.4      | 0.4     | Flow error threshold                  |
| `cellprob_threshold`  | 0.0      | 0.0     | Cell probability threshold            |
| `tile_norm_blocksize` | 0        | 0       | Block size for tile normalisation (0 = global) |
| `resize`              | 0        | 0       | Resize longest side before segmentation (0 = no resize) |
| `min_cell_size`       | 15       | 15      | Minimum cell size in pixels           |

### Model (`src/cell_size/configs/model/cpsam.yaml`)

| Key                 | Default  | Description                              |
|---------------------|----------|------------------------------------------|
| `model_type`        | `"cpsam"` | Cellpose model name                     |
| `custom_model_path` | `null`   | Path to a custom fine-tuned model        |
| `gpu`               | `true`   | Use GPU (falls back to CPU if unavailable) |
| `batch_size`        | `32`     | Batch size for model evaluation          |

### Output (`src/cell_size/configs/output/default.yaml`)

| Key                  | Default        | Description                                  |
|----------------------|----------------|----------------------------------------------|
| `mask_format`        | `"tif"`        | `"tif"` or `"npy"`                           |
| `csv_path`           | `"results.csv"` | Catalog CSV file path (relative to data_dir) |
| `generate_overlays`  | `false`        | Save outline overlay PNGs                    |
| `generate_plots`     | `false`        | Save cell-area histogram PNGs                |
| `compute_cell_areas` | `false`        | Write per-image cell area CSVs               |

## Output Structure

After processing, each image gets its own folder:

```
data_dir/
  projectA/
    image000/
      image000.jpg            # original image (moved here)
      image000_mask.tif       # segmentation mask (uint16)
      image000_overlay.png    # (optional) outline overlay
      image000_areas.csv      # (optional) per-cell areas
      image000_histogram.png  # (optional) area distribution
    image001/
      image001.tif
      image001_mask.tif
  results.csv                 # catalog CSV
```

### Catalog CSV format

```csv
Relative_Path,Image_Name,File_Type,Mask_Name,Resize,Cell_Type,Timestamp
projectA/image000,image000,jpg,image000_mask,1000,FrogBlood,2026-03-10T17:30:00
```

## License

MIT
