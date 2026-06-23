"""Process a single Sentinel-2 L1C scene into water-quality products.

Runs ACOLITE atmospheric correction (L1C ``.SAFE`` -> L2W) followed by
MoE-VAE inference on one Sentinel-2 scene and writes validated Cloud Optimized
GeoTIFFs (chl-a, TSS, aCDOM) plus a merged products NetCDF.

Examples::

    python run_file.py S2B_MSIL1C_20241024T160239_..._T17RLL_....SAFE
    python run_file.py /path/to/scene.SAFE --output /path/to/output

ACOLITE must be installed locally or it is downloaded automatically; point at
an existing install via the ``ACOLITE_DIR`` environment variable (or
``--acolite-dir``).

To process every scene in a folder, use ``run_folder.py``.
"""

import os
import argparse

import torch

from s2_processing import BASE_DIR, load_models, process_scene

# Default to the shared data drive; fall back to local folders next to the
# scripts if the drive is not mounted.
DEFAULT_ROOT = "/media/hdd/Data/S2"
DEFAULT_DATA = os.path.join(DEFAULT_ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_ROOT, "output")
DEFAULT_L2 = os.path.join(DEFAULT_ROOT, "L2")

parser = argparse.ArgumentParser(
    description="Process a single Sentinel-2 L1C scene into water-quality " "products."
)
parser.add_argument(
    "input",
    help="Input Sentinel-2 .SAFE directory. Either a path, or a name in the "
    "data folder.",
)
parser.add_argument(
    "--model-dir",
    default=os.path.join(BASE_DIR, "model"),
    help="Directory containing the model subfolders (default: ./model).",
)
parser.add_argument(
    "--output",
    default=DEFAULT_OUTPUT,
    help=f"Output directory for the products (default: {DEFAULT_OUTPUT}).",
)
parser.add_argument(
    "--l2-dir",
    default=DEFAULT_L2,
    help=f"Directory for intermediate ACOLITE L2 output (default: {DEFAULT_L2}).",
)
parser.add_argument(
    "--acolite-dir",
    default=None,
    help="ACOLITE install directory (default: $ACOLITE_DIR; downloaded "
    "automatically if missing).",
)
parser.add_argument(
    "--no-download",
    action="store_true",
    help="Do not download ACOLITE if it is missing (error instead).",
)
args = parser.parse_args()

# Resolve the input path: use it as given if it exists, otherwise look in the
# default data folder.
if os.path.exists(args.input):
    safe_path = os.path.abspath(args.input.rstrip("/"))
else:
    safe_path = os.path.join(DEFAULT_DATA, args.input)
if not os.path.exists(safe_path):
    raise FileNotFoundError(f"Input .SAFE not found: {args.input}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = load_models(args.model_dir, device)
process_scene(
    safe_path,
    models,
    save_dir=args.output,
    l2_dir=args.l2_dir,
    acolite_dir=args.acolite_dir,
    download=not args.no_download,
)
print("Done.")
