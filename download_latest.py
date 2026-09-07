"""Download the latest Sentinel-2 L1C scene(s) for the workflow.

Queries the Copernicus Data Space Ecosystem (CDSE) catalogue over a region of
interest, walking back through progressively larger time windows until scenes
are found, and downloads the most recent ``.SAFE`` product(s).

For downloading scenes over a specific date range instead, use
``download_data.py``.

A free CDSE account is required for the download step. Provide credentials via
the ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` environment variables (register at
https://dataspace.copernicus.eu).
"""

import os
from datetime import datetime, timedelta, timezone

from s2_cdse import download_s2, search_s2_l1c

# Default to the shared data drive; fall back to a local folder if unmounted.
DATA_DIR = os.environ.get("S2_DATA_DIR", "/media/hdd/Data/S2/data")
os.makedirs(DATA_DIR, exist_ok=True)

# Bounding box over the Gulf of Mexico / U.S. Gulf coast, matching the region
# covered by the bundled test scene. [xmin, ymin, xmax, ymax]
BBOX = (-98.0, 18.0, -80.0, 31.0)
MAX_CLOUD = 80.0  # maximum scene cloud cover percent
NUM_SCENES = 1  # number of most-recent scenes to download
# Look-back windows (days) tried in order until scenes are found.
LOOKBACK_DAYS = (7, 30, 90, 365)

end = datetime.now(timezone.utc)
scenes = []
for days in LOOKBACK_DAYS:
    start = end - timedelta(days=days)
    print(f"Searching {start:%Y-%m-%d} to {end:%Y-%m-%d} ...")
    scenes = search_s2_l1c(
        bbox=BBOX,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        max_cloud=MAX_CLOUD,
    )
    if scenes:
        break

if not scenes:
    raise RuntimeError(
        f"No Sentinel-2 L1C scenes found over the bounding box in the last "
        f"{LOOKBACK_DAYS[-1]} days."
    )

scenes = scenes[:NUM_SCENES]
print(f"\nLatest {len(scenes)} scene(s):")
for s in scenes:
    print("  ", s["name"], f"(cloud {s['cloud']:.0f}%)")

files = download_s2(scenes, out_dir=DATA_DIR)
print(f"\nDownloaded {len(files)} scene(s) to {DATA_DIR}:")
for f in files:
    print("  ", os.path.basename(f))
