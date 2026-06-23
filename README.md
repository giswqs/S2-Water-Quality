# Sentinel-2 Water-Quality Products (MoE-VAE)

Generate gridded water-quality products from ESA **Sentinel-2 MSI** imagery
using a Mixture-of-Experts Variational Autoencoder (MoE-VAE). Sentinel-2 is
distributed as **L1C top-of-atmosphere** `.SAFE` products, so each scene is
first atmospherically corrected with
[**ACOLITE**](https://github.com/acolite/acolite) (the `radcor` method) to
produce an L2W surface-reflectance (`Rrs`) product, then run through the
models. For each scene the workflow estimates three products and writes both a
multi-variable NetCDF and validated Cloud Optimized GeoTIFFs (COGs):

| Product   | Variable    | Units      |
|-----------|-------------|------------|
| Chlorophyll-a | `chla`      | mg m⁻³ |
| Total Suspended Solids | `tss`       | g m⁻³  |
| CDOM absorption @ 440 nm | `acdom440`  | m⁻¹    |

## Pipeline

```
Sentinel-2 L1C (.SAFE)  ──ACOLITE──►  L2W Rrs (.nc)  ──MoE-VAE──►  chla / tss / acdom
       download_*.py                  run_acolite                COGs + NetCDF
```

## Project layout

```
S2-Water-Quality/
├── code/                  # MoE-VAE model + inference/IO helpers
├── model/                 # Trained weights & scalers (Chl-a / TSS / aCDOM440)
├── download_data.py       # Download L1C scenes over a specified date range
├── download_latest.py     # Download the most recent L1C scene
├── s2_cdse.py             # Copernicus Data Space search/download helpers
├── s2_processing.py       # Shared logic: ACOLITE / load_models / process_scene / COG
├── run_file.py            # Process a single scene (ACOLITE + inference)
├── run_folder.py          # Process every scene in a folder (best pass per day)
├── requirements.txt
└── README.md
```

By default, downloads, intermediate ACOLITE L2 output, and products are written
to the shared data drive:

```
/media/hdd/Data/S2/
├── data/      # input Sentinel-2 .SAFE scenes
├── L2/        # intermediate ACOLITE L2R / L2W output (per scene)
└── output/    # generated products (COGs + NetCDF)
```

Override these with `--output`, `--l2-dir`, and the `S2_DATA_DIR` environment
variable.

## Installation

Python 3.10+ with a CUDA-capable GPU recommended (CPU works but is slower).

```bash
pip install -r requirements.txt
```

### ACOLITE

The atmospheric-correction step uses **ACOLITE** (which ships its own bundled
Python and is *not* pip-installable). HyperCoast can fetch and run it for you
(`hypercoast.download_acolite` / `hypercoast.run_acolite`), so the run scripts
download ACOLITE automatically on first use.

> **Note:** the Sentinel-2 settings use the `radcor` atmospheric-correction
> method, which requires a **recent ACOLITE release (2024 or newer)**.
> HyperCoast's auto-download fetches an older release that predates `radcor`,
> so point `ACOLITE_DIR` at a recent install:
>
> ```bash
> export ACOLITE_DIR=/path/to/acolite_py_linux_2025
> ```

Pass `--acolite-dir` to point at a specific install, or `--no-download` to
fail instead of fetching ACOLITE when it is missing.

## Copernicus Data Space credentials

Sentinel-2 L1C `.SAFE` products are distributed by ESA through the
[Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu)
(not NASA Earthdata). The catalogue search is open, but downloading requires a
free CDSE account. Provide credentials via environment variables:

```bash
export CDSE_USERNAME=you@example.com
export CDSE_PASSWORD=your_password
```

## Usage

### 1. Download data

```bash
# Latest available scene over the region of interest
python download_latest.py

# Scenes over a specific date range (Gulf of Mexico by default)
python download_data.py 2024-10-01 2024-10-31
python download_data.py 2024-10-01 2024-10-31 --count 5
python download_data.py 2024-10-24 2024-10-24 --bbox -99 18 -78 42
```

### 2. Process scenes

```bash
# A single scene (path, or a .SAFE name found in the data folder)
python run_file.py S2B_MSIL1C_20241024T160239_..._T17RLL_....SAFE
python run_file.py /path/to/scene.SAFE --output /path/to/output

# Every scene in a folder (defaults to /media/hdd/Data/S2/data -> .../output)
python run_folder.py
python run_folder.py /path/to/scenes --output /path/to/output
```

Both run ACOLITE first (writing intermediate L2W into `--l2-dir`) and then
inference. `run_folder.py` loads the models once, keeps the **best pass per
day** (most valid retrieval pixels), and continues past individual scene
failures (reporting them in a summary).

## Outputs

For a scene acquired on `<YYYYMMDD>`, the output folder receives one
date-named COG per product plus a merged NetCDF:

- `S2-<YYYYMMDD>-chla.tif`
- `S2-<YYYYMMDD>-tss.tif`
- `S2-<YYYYMMDD>-acdom.tif`
- `S2-<YYYYMMDD>-products.nc`

The date is parsed from the input filename. When several Sentinel-2 passes
share a date, the pass with the most valid retrieval pixels is kept (best pass
per day).

### About the COGs

The L2W product is georeferenced by 2D `lat`/`lon` arrays, so each product is
gridded at its true `(lon, lat)` onto a regular EPSG:4326 grid (~100 m,
0.001°) with `scipy.interpolate.griddata`. This georeferences correctly while
leaving masked / no-data areas as nodata. Each GeoTIFF is written with internal
tiling, overviews and DEFLATE compression, then validated with `rio_cogeo`.

Inference is deterministic: the models run in `eval()` mode, which disables
the MoE noisy gating and makes the VAE use its latent mean, so re-running a
scene reproduces the same products.

### Bands

Each product uses a different subset of Sentinel-2 MSI bands (the closest L2W
`Rrs` band to each target wavelength is selected at inference time):

- **Chl-a**: 443–740 nm (row-wise min-max normalization)
- **TSS**: 443–865 nm (robust scaler)
- **aCDOM440**: 443–704 nm (robust scaler)

## Automated daily products

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs every day
(and on demand via *Run workflow*). It downloads the most recent Sentinel-2
scene, runs ACOLITE + inference, and publishes the resulting GeoTIFFs to two
places:

- the repository's **`S2-Data`** release
- the **Hugging Face dataset** (under `cogs/`):
  https://huggingface.co/datasets/giswqs/S2-Water-Quality

Because output filenames include the acquisition date, products from different
dates accumulate while same-date files are replaced.

The workflow targets a **self-hosted runner** because it depends on a recent
ACOLITE install (radcor) and processes Sentinel-2 `.SAFE` products onto the
local data drive. Configure `ACOLITE_DIR` as a repository variable and set
these secrets under **Settings → Secrets and variables → Actions**:

- `CDSE_USERNAME` — Copernicus Data Space login
- `CDSE_PASSWORD` — Copernicus Data Space password
- `HF_TOKEN` — Hugging Face token with write access to the dataset

## Notes

- The `data/`, `L2/` and `output/` folders (and the ACOLITE install) are
  git-ignored, so large scenes and products are never committed.
- The `model/` weights are required and bundled in the repository.
