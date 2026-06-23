"""Download Sentinel-2 L1C scenes over a specified date range.

Sentinel-2 L1C ``.SAFE`` products are distributed by ESA through the
**Copernicus Data Space Ecosystem (CDSE)**, not NASA Earthdata, so this script
queries the CDSE STAC catalogue and downloads via the CDSE OData/zipper API.

For downloading only the most recent scene instead, use ``download_latest.py``.

Examples::

    python download_data.py 2024-10-01 2024-10-31
    python download_data.py 2024-10-01 2024-10-31 --count 5
    python download_data.py 2024-10-24 2024-10-24 --bbox -99 18 -78 42

A free CDSE account is required for the download step. Provide credentials via
the ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` environment variables (register at
https://dataspace.copernicus.eu). The catalogue search itself needs no login.
"""

import os
import argparse

from s2_cdse import search_s2_l1c, download_s2

# Default to the shared data drive; fall back to a local folder if unmounted.
DEFAULT_DATA_DIR = os.environ.get("S2_DATA_DIR", "/media/hdd/Data/S2/data")

# Default bounding box over the Gulf of Mexico / U.S. Gulf coast, matching the
# region covered by the bundled test scene. [xmin, ymin, xmax, ymax]
DEFAULT_BBOX = (-98.0, 18.0, -80.0, 31.0)

parser = argparse.ArgumentParser(
    description="Download Sentinel-2 L1C scenes over a specified date range."
)
parser.add_argument("start", help="Start date (YYYY-MM-DD).")
parser.add_argument("end", help="End date (YYYY-MM-DD), inclusive.")
parser.add_argument(
    "--count",
    type=int,
    default=3,
    help="Maximum number of scenes to download (default: 3). Use -1 for all.",
)
parser.add_argument(
    "--bbox",
    type=float,
    nargs=4,
    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    default=DEFAULT_BBOX,
    help="Bounding box (default: Gulf of Mexico).",
)
parser.add_argument(
    "--max-cloud",
    type=float,
    default=80.0,
    help="Maximum scene cloud cover percent (default: 80).",
)
parser.add_argument(
    "--out-dir",
    default=DEFAULT_DATA_DIR,
    help=f"Directory to download into (default: {DEFAULT_DATA_DIR}).",
)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

print(f"Searching {args.start} to {args.end} over {tuple(args.bbox)} ...")
scenes = search_s2_l1c(
    bbox=tuple(args.bbox),
    start=args.start,
    end=args.end,
    max_cloud=args.max_cloud,
)
if args.count != -1:
    scenes = scenes[: args.count]

if not scenes:
    raise RuntimeError(
        f"No Sentinel-2 L1C scenes found over {tuple(args.bbox)} between "
        f"{args.start} and {args.end}."
    )

print(f"Found {len(scenes)} scene(s):")
for s in scenes:
    print("  ", s["name"], f"(cloud {s['cloud']:.0f}%)")

files = download_s2(scenes, out_dir=args.out_dir)
print(f"\nDownloaded {len(files)} scene(s) to {args.out_dir}:")
for f in files:
    print("  ", os.path.basename(f))
