"""Copernicus Data Space Ecosystem (CDSE) search/download helpers for S2 L1C.

Sentinel-2 L1C ``.SAFE`` products are an ESA product, distributed through the
Copernicus Data Space Ecosystem rather than NASA Earthdata. These helpers wrap
the CDSE OData catalogue (open, no login for search) and the CDSE zipper
download endpoint (requires a free CDSE account).

* :func:`search_s2_l1c` - query L1C scenes by bbox / date / cloud cover.
* :func:`download_s2` - download and unzip scenes to ``.SAFE`` directories.

Credentials for the download step are read from the ``CDSE_USERNAME`` /
``CDSE_PASSWORD`` environment variables (register at
https://dataspace.copernicus.eu).
"""

import os
import zipfile

import requests

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


def _bbox_to_wkt(bbox):
    """Convert an (xmin, ymin, xmax, ymax) bbox to a closed WKT polygon.

    Args:
        bbox (tuple): ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.

    Returns:
        str: A WKT ``POLYGON`` string spanning the bbox.
    """
    xmin, ymin, xmax, ymax = bbox
    return (
        f"POLYGON(({xmin} {ymin},{xmax} {ymin},{xmax} {ymax},"
        f"{xmin} {ymax},{xmin} {ymin}))"
    )


def search_s2_l1c(bbox, start, end, max_cloud=80.0, max_records=100):
    """Search Sentinel-2 L1C scenes in the CDSE catalogue.

    Args:
        bbox (tuple): ``(xmin, ymin, xmax, ymax)`` bounding box (EPSG:4326).
        start (str): Start date ``YYYY-MM-DD`` (inclusive).
        end (str): End date ``YYYY-MM-DD`` (inclusive).
        max_cloud (float): Maximum scene cloud-cover percent (default 80).
        max_records (int): Maximum catalogue records to request (default 100).

    Returns:
        list[dict]: Scenes sorted newest-first, each
            ``{"id", "name", "cloud", "date"}``.
    """
    wkt = _bbox_to_wkt(bbox)
    flt = (
        "Collection/Name eq 'SENTINEL-2' "
        "and contains(Name,'MSIL1C') "
        f"and ContentDate/Start gt {start}T00:00:00.000Z "
        f"and ContentDate/Start lt {end}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
        "and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq "
        f"'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
    )
    params = {
        "$filter": flt,
        "$orderby": "ContentDate/Start desc",
        "$top": str(max_records),
    }
    resp = requests.get(CATALOGUE_URL, params=params, timeout=120)
    resp.raise_for_status()

    scenes = []
    for item in resp.json().get("value", []):
        cloud = next(
            (
                a["Value"]
                for a in item.get("Attributes", [])
                if a.get("Name") == "cloudCover"
            ),
            float("nan"),
        )
        scenes.append(
            {
                "id": item["Id"],
                "name": item["Name"],
                "cloud": cloud,
                "date": item.get("ContentDate", {}).get("Start", ""),
            }
        )
    return scenes


def _get_access_token():
    """Obtain a CDSE access token from CDSE_USERNAME / CDSE_PASSWORD.

    Returns:
        str: A bearer access token.

    Raises:
        RuntimeError: If credentials are not set in the environment.
    """
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "CDSE credentials required to download. Set CDSE_USERNAME and "
            "CDSE_PASSWORD (register at https://dataspace.copernicus.eu)."
        )
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_s2(scenes, out_dir, unzip=True):
    """Download Sentinel-2 scenes from CDSE and (optionally) unzip to .SAFE.

    Args:
        scenes (list[dict]): Scenes from :func:`search_s2_l1c`.
        out_dir (str): Destination directory.
        unzip (bool): Unzip each product to a ``.SAFE`` directory and remove
            the zip (default True).

    Returns:
        list[str]: Paths to the downloaded ``.SAFE`` directories (or zips when
            ``unzip`` is False).
    """
    os.makedirs(out_dir, exist_ok=True)
    token = _get_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    paths = []
    for scene in scenes:
        safe_dir = os.path.join(out_dir, scene["name"])
        if os.path.exists(safe_dir):
            print(f"  {scene['name']} already present, skipping.")
            paths.append(safe_dir)
            continue

        zip_path = os.path.join(out_dir, scene["name"].replace(".SAFE", "") + ".zip")
        url = f"{ZIPPER_URL}({scene['id']})/$value"
        print(f"  downloading {scene['name']} ...")
        with requests.get(url, headers=headers, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

        if unzip:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
            os.remove(zip_path)
            paths.append(safe_dir if os.path.exists(safe_dir) else out_dir)
        else:
            paths.append(zip_path)
    return paths
