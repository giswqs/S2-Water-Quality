"""Shared Sentinel-2 water-quality processing logic.

This module holds the reusable pieces of the Sentinel-2 MSI MoE-VAE inference
workflow so that the model weights only have to be loaded once and can be
reused across many scenes. Sentinel-2 is distributed as L1C top-of-atmosphere
``.SAFE`` products, so each scene is first atmospherically corrected with
**ACOLITE** to produce an L2W ``Rrs`` product before inference:

* :func:`run_acolite` - L1C ``.SAFE`` -> L2W surface reflectance (``Rrs_*``).
* :func:`load_models` - build the chl-a / TSS / aCDOM models and scalers.
* :func:`infer_scene_maps` - run inference on one L2W scene (in memory).
* :func:`save_product_to_cog` - grid a swath product and write a valid COG.
* :func:`save_products_to_nc` - write a merged multi-variable NetCDF.
* :func:`process_scene` - end-to-end ACOLITE + inference + outputs.

Entry points:

* ``run_file.py`` processes a single ``.SAFE`` scene.
* ``run_folder.py`` processes every ``.SAFE`` scene in a folder.
"""

import os
import re
import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import torch
import hypercoast
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

# Resolve paths relative to this module so it can run from any location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "code"))

from MoE_VAE import *  # noqa: E402,F401,F403
from data_loading import *  # noqa: E402,F401,F403
from plot_and_save import *  # noqa: E402,F401,F403
from model_inference import (  # noqa: E402
    preprocess_and_infer_emit_minmax,
    preprocess_and_infer_emit_robust,
)

# ===========================================================================
# Band definitions (nm). Sentinel-2 MSI has a handful of discrete bands; the
# closest L2W ``Rrs`` band to each target wavelength is selected at inference
# time. Each product uses a different subset, matching how the models were
# trained.
# ===========================================================================
BANDS_443_704 = [443, 492, 560, 665, 704]
BANDS_443_740 = BANDS_443_704 + [740]
BANDS_443_865 = BANDS_443_740 + [783, 833, 865]

# Per-product band selection (chl-a / TSS / aCDOM use different subsets).
PRODUCT_BANDS = {
    "chla": BANDS_443_740,
    "tss": BANDS_443_865,
    "acdom440": BANDS_443_704,
}

# Map dataset variable -> output filename label (aCDOM drops the "440").
PRODUCT_LABELS = {"chla": "chla", "tss": "tss", "acdom440": "acdom"}

# ===========================================================================
# ACOLITE configuration. Atmospheric correction is run through HyperCoast's
# ``run_acolite`` (which locates the bundled ``dist/acolite/acolite``
# executable) and HyperCoast's ``download_acolite`` (which fetches a complete
# official release). Point ACOLITE_DIR at an existing install to skip the
# download; otherwise it is downloaded on first use.
# ===========================================================================
# `or` (not a get default) so an empty ACOLITE_DIR env var still falls back.
ACOLITE_DIR = os.environ.get("ACOLITE_DIR") or os.path.join(
    BASE_DIR, "acolite_py_linux"
)

# Sentinel-2-specific ACOLITE processing settings (radcor atmospheric
# correction + SWIR water mask), appended to the per-scene
# ``inputfile``/``output`` lines.
S2_ACOLITE_SETTINGS = """\
polygon=None
limit=None
atmospheric_correction=True
atmospheric_correction_method=radcor
l2w_parameters=Rrs_*
rgb_rhot=True
rgb_rhos=True
map_l2w=False
l2w_mask_wave=1600
l2w_mask_threshold=0.1
"""


def _acolite_binary(acolite_dir):
    """Return the path to the ACOLITE executable inside an install dir.

    Args:
        acolite_dir (str): The extracted ACOLITE directory (e.g.
            ``.../acolite_py_linux``).

    Returns:
        str: Path to the ``dist/acolite/acolite`` (or ``acolite.exe``) binary.
    """
    exe = "acolite.exe" if acolite_dir.rstrip("/\\").endswith("win") else "acolite"
    return os.path.join(acolite_dir, "dist", "acolite", exe)


def ensure_acolite(acolite_dir=None, download=True):
    """Return a usable ACOLITE install dir, downloading one if necessary.

    Args:
        acolite_dir (str, optional): Candidate ACOLITE install directory.
            Defaults to ``ACOLITE_DIR``.
        download (bool): If the install is missing, fetch a complete official
            release with ``hypercoast.download_acolite`` (default True).

    Returns:
        str: A directory whose ``dist/acolite/acolite`` executable exists.

    Raises:
        FileNotFoundError: If the install is missing and ``download`` is False.
    """
    acolite_dir = acolite_dir or ACOLITE_DIR
    if os.path.exists(_acolite_binary(acolite_dir)):
        return acolite_dir
    if not download:
        raise FileNotFoundError(
            f"ACOLITE executable not found under {acolite_dir}. Set ACOLITE_DIR "
            "to an existing install, or allow download=True."
        )
    # download_acolite extracts to <outdir>/acolite_py_<os> and returns it.
    outdir = os.path.dirname(os.path.abspath(acolite_dir)) or "."
    print("ACOLITE not found; downloading a release ...")
    return hypercoast.download_acolite(outdir=outdir)


# Conda/GDAL environment variables that would otherwise leak into the ACOLITE
# subprocess. ACOLITE is a self-contained PyInstaller bundle with its own GDAL,
# PROJ and shared libraries; if it inherits the conda env's GDAL_DRIVER_PATH /
# LD_LIBRARY_PATH it loads conda's libgdal (which needs a newer libtiff than
# ACOLITE bundles) and fails. These are stripped for the subprocess only.
_ACOLITE_STRIP_ENV = (
    "GDAL_DRIVER_PATH",
    "GDAL_DATA",
    "GDAL_PLUGIN_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
    "PROJ_NETWORK",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)


class _clean_acolite_env:
    """Temporarily strip conda/GDAL env vars so ACOLITE uses its own bundle.

    The vars are removed from ``os.environ`` on enter (so a subprocess spawned
    inside the block inherits a clean environment) and restored on exit.
    Already-loaded libraries in the current process are unaffected.
    """

    def __enter__(self):
        self._saved = {
            k: os.environ.pop(k) for k in _ACOLITE_STRIP_ENV if k in os.environ
        }
        return self

    def __exit__(self, *exc):
        os.environ.update(self._saved)
        return False


def run_acolite(safe_path, out_root, acolite_dir=None, download=True):
    """Atmospherically correct one Sentinel-2 L1C ``.SAFE`` scene with ACOLITE.

    ACOLITE reads the L1C ``.SAFE`` product and writes L2R / L2W products,
    including the ``Rrs_*`` surface-reflectance bands the models consume. The
    run is driven by a Sentinel-2-tuned settings file (radcor atmospheric
    correction) and executed through ``hypercoast.run_acolite``.

    Args:
        safe_path (str): Path to the Sentinel-2 ``.SAFE`` product directory.
        out_root (str): Root directory for ACOLITE output (a per-scene
            subfolder is created inside it).
        acolite_dir (str, optional): ACOLITE install directory. Defaults to
            ``ACOLITE_DIR``; downloaded automatically if missing.
        download (bool): Download ACOLITE if it is not already installed
            (default True).

    Returns:
        dict: ``{"scene", "input": safe_path, "output_dir", "l2r_files",
            "l2w_files"}``.

    Raises:
        FileNotFoundError: If ACOLITE produces no L2W file.
    """
    acolite_dir = ensure_acolite(acolite_dir, download=download)

    scene = Path(safe_path).stem
    out_path = os.path.join(out_root, scene)
    os.makedirs(out_path, exist_ok=True)

    settings_file = os.path.join(out_path, f"{scene}_acolite_settings.txt")
    with open(settings_file, "w") as f:
        f.write(f"inputfile={safe_path}\noutput={out_path}\n{S2_ACOLITE_SETTINGS}")

    print("=" * 80)
    print(f"Running ACOLITE for Sentinel-2 SAFE: {safe_path}")
    print("=" * 80)
    # out_dir is passed explicitly: hypercoast.run_acolite requires it even
    # when a settings file is supplied. The clean-env block keeps conda's GDAL
    # out of ACOLITE's bundled runtime.
    with _clean_acolite_env():
        hypercoast.run_acolite(
            acolite_dir, settings_file=settings_file, out_dir=out_path
        )

    l2w_files = sorted(glob.glob(os.path.join(out_path, "*L2W*.nc")))
    l2r_files = sorted(glob.glob(os.path.join(out_path, "*L2R*.nc")))
    if not l2w_files:
        raise FileNotFoundError(f"No L2W file generated in: {out_path}")

    print(f"ACOLITE finished. L2W: {l2w_files[0]}")
    return {
        "scene": scene,
        "input": str(safe_path),
        "output_dir": out_path,
        "l2r_files": l2r_files,
        "l2w_files": l2w_files,
    }


def _build_model(
    input_dim, encoder_hidden_dims, decoder_hidden_dims, use_softplus_output, device
):
    """Construct a MoE-VAE with the Sentinel-2 product architecture.

    Args:
        input_dim (int): Number of input spectral bands.
        encoder_hidden_dims (list[int]): Encoder hidden layer widths.
        decoder_hidden_dims (list[int]): Decoder hidden layer widths.
        use_softplus_output (bool): Whether to apply a softplus on the output
            (used for the chl-a model).
        device (torch.device): Device to place the model on.

    Returns:
        MoE_VAE: The constructed (untrained) model on ``device``.
    """
    return MoE_VAE(  # noqa: F405
        input_dim=input_dim,
        output_dim=1,
        latent_dim=16,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        activation="leakyrelu",
        use_norm="layer",
        use_dropout=False,
        use_softplus_output=use_softplus_output,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ).to(device)


def load_models(model_dir, device):
    """Build the chl-a, TSS and aCDOM models and load their weights/scalers.

    Args:
        model_dir (str): Directory containing the ``Chl-a``, ``TSS`` and
            ``aCDOM440`` model subfolders.
        device (torch.device): Device to load the models onto.

    Returns:
        dict: Mapping of product name to a dict with the loaded ``model`` and,
            for TSS/aCDOM, the ``scaler_Rrs`` and ``scaler_dict`` objects.
    """
    chla_model = _build_model(
        input_dim=len(PRODUCT_BANDS["chla"]),
        encoder_hidden_dims=[128, 64, 32],
        decoder_hidden_dims=[32, 64, 128],
        use_softplus_output=True,
        device=device,
    )
    chla_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "Chl-a", "best_model_minloss.pth"),
            map_location=device,
        )
    )

    tss_model = _build_model(
        input_dim=len(PRODUCT_BANDS["tss"]),
        encoder_hidden_dims=[128, 64, 32],
        decoder_hidden_dims=[32, 64, 128],
        use_softplus_output=False,
        device=device,
    )
    tss_dir = os.path.join(model_dir, "TSS")
    tss_model.load_state_dict(
        torch.load(os.path.join(tss_dir, "best_model_minloss.pth"), map_location=device)
    )
    with open(os.path.join(tss_dir, "scalers_Rrs_real.pkl"), "rb") as f:
        tss_scaler_Rrs = pickle.load(f)
    tss_scaler_dict = torch.load(
        os.path.join(tss_dir, "scaler.pt"), map_location="cpu", weights_only=False
    )

    acdom_model = _build_model(
        input_dim=len(PRODUCT_BANDS["acdom440"]),
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        use_softplus_output=False,
        device=device,
    )
    acdom_dir = os.path.join(model_dir, "aCDOM440")
    acdom_model.load_state_dict(
        torch.load(
            os.path.join(acdom_dir, "best_model_minloss.pth"), map_location=device
        )
    )
    with open(os.path.join(acdom_dir, "scalers_Rrs_real.pkl"), "rb") as f:
        acdom_scaler_Rrs = pickle.load(f)
    acdom_scaler_dict = torch.load(
        os.path.join(acdom_dir, "scaler.pt"), map_location="cpu", weights_only=False
    )

    # eval() mode: disables noisy gating and makes the VAE use the latent
    # mean (deterministic inference).
    for mdl in (chla_model, tss_model, acdom_model):
        mdl.eval()

    return {
        "chla": {"model": chla_model},
        "tss": {
            "model": tss_model,
            "scaler_Rrs": tss_scaler_Rrs,
            "scaler_dict": tss_scaler_dict,
        },
        "acdom440": {
            "model": acdom_model,
            "scaler_Rrs": acdom_scaler_Rrs,
            "scaler_dict": acdom_scaler_dict,
        },
    }


def save_product_to_cog(
    out_tif,
    lat_2d,
    lon_2d,
    values_2d,
    resolution_m=100,
    method="linear",
    nodata=-9999.0,
):
    """Grid a Sentinel-2 swath product onto a regular grid and write a COG.

    The L2W product is georeferenced by 2D ``lat``/``lon`` arrays. Each pixel
    is gridded at its true (lon, lat) onto a regular EPSG:4326 grid with
    ``scipy.interpolate.griddata``, which georeferences correctly and leaves
    masked / no-data areas as nodata. The result is written as a Cloud
    Optimized GeoTIFF (internal tiling, overviews, DEFLATE compression) and
    validated.

    Args:
        out_tif (str): Output GeoTIFF path.
        lat_2d (np.ndarray): Latitude (degrees north, EPSG:4326).
        lon_2d (np.ndarray): Longitude (degrees east, EPSG:4326).
        values_2d (np.ndarray): Product values aligned with lat/lon (NaN for
            invalid pixels).
        resolution_m (float): Target grid resolution in metres (default 100,
            ~0.001 deg; Sentinel-2 water-quality bands are 10-60 m native).
        method (str): ``griddata`` interpolation method (default "linear").
        nodata (float): Value used for empty cells.

    Returns:
        str: The path to the validated COG.
    """
    from scipy.interpolate import griddata

    lat = np.asarray(lat_2d, dtype=np.float64).ravel()
    lon = np.asarray(lon_2d, dtype=np.float64).ravel()
    val = np.asarray(values_2d, dtype=np.float64).ravel()

    geo_ok = np.isfinite(lat) & np.isfinite(lon)
    if not (geo_ok & np.isfinite(val)).any():
        raise ValueError(f"No valid pixels to grid for {out_tif}")
    lat, lon, val = lat[geo_ok], lon[geo_ok], val[geo_ok]

    # Regular grid spanning the swath extent; metres -> degrees at scene centre.
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_c = (lat_min + lat_max) / 2.0
    res_lat = resolution_m / 111000.0
    res_lon = resolution_m / (111000.0 * np.cos(np.radians(lat_c)))
    lon_axis = np.arange(lon_min, lon_max + res_lon, res_lon)
    lat_axis = np.arange(lat_min, lat_max + res_lat, res_lat)
    mesh_lon, mesh_lat = np.meshgrid(lon_axis, lat_axis)
    transform = from_origin(lon_axis.min(), lat_axis.max(), res_lon, res_lat)

    # Grid by true (lon, lat); NaN-valued pixels keep gaps where there is no
    # data (interpolation does not cross them).
    grid = griddata((lon, lat), val, (mesh_lon, mesh_lat), method=method)
    grid = np.flipud(grid).astype(np.float32)
    filled = np.isfinite(grid)
    grid[~filled] = nodata

    nrow, ncol = grid.shape
    src_profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=nrow,
        width=ncol,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    )
    dst_profile = cog_profiles.get("deflate")
    with MemoryFile() as mem:
        with mem.open(**src_profile) as src:
            src.write(grid, 1)
        with mem.open() as src:
            cog_translate(
                src,
                out_tif,
                dst_profile,
                overview_resampling="nearest",
                quiet=True,
            )

    is_valid, errors, warnings = cog_validate(out_tif)
    status = "valid" if is_valid else "INVALID"
    print(
        f"COG {status}: {out_tif} "
        f"({int(filled.sum())} cells @ {res_lon:.5f}x{res_lat:.5f} deg)"
    )
    if errors:
        print("  errors:", errors)
    if warnings:
        print("  warnings:", warnings)
    return out_tif


def parse_acquisition_date(path):
    """Parse the acquisition date (YYYYMMDD) from an S2/ACOLITE filename.

    Handles both the Sentinel-2 L1C ``.SAFE`` name
    (``S2B_MSIL1C_20241024T160239_N0511_R097_T17RLL_...``) and the ACOLITE
    L2W output name (``S2B_MSI_2024_10_24_16_15_47_T17RLL_L2W.nc``).

    Args:
        path (str): Path to a Sentinel-2 ``.SAFE`` or ACOLITE L2W file.

    Returns:
        str: The 8-digit date string (e.g. ``"20241024"``).

    Raises:
        ValueError: If no date can be parsed from the filename.
    """
    name = os.path.basename(path.rstrip("/\\"))
    match = re.search(r"_(\d{8})T\d{6}", name)
    if match:
        return match.group(1)
    match = re.search(r"_(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}", name)
    if match:
        return "".join(match.group(1, 2, 3))
    raise ValueError(f"Could not parse acquisition date from: {path}")


def infer_scene_maps(nc_path, models):
    """Run inference on one Sentinel-2 L2W scene and return product maps.

    No files are written. The per-pixel model outputs are reshaped to the
    scene's native grid so they can be gridded directly to GeoTIFFs.

    Args:
        nc_path (str): Path to the ACOLITE L2W NetCDF file (with ``Rrs_*``).
        models (dict): Loaded models/scalers from :func:`load_models`.

    Returns:
        dict: ``{"latitude", "longitude", "chla", "tss", "acdom440",
            "valid"}`` where the first five are 2D arrays and ``valid`` is the
            number of valid (finite) chl-a retrieval pixels.
    """
    # Chl-a uses the row-wise min-max model; returns a flat [lat, lon, value].
    # Sentinel-2 has few bands, so the ascending spectral mask is disabled.
    chla_flat = preprocess_and_infer_emit_minmax(
        nc_path=nc_path,
        model=models["chla"]["model"],
        full_band_wavelengths=PRODUCT_BANDS["chla"],
        use_spectral_mask=False,
    )

    # TSS / aCDOM use the robust model; return 2D maps plus the geolocation.
    tss_flat, tss_2d, _, lat, lon = preprocess_and_infer_emit_robust(
        nc_path=nc_path,
        model=models["tss"]["model"],
        scaler_Rrs=models["tss"]["scaler_Rrs"],
        TSS_scalers_dict=models["tss"]["scaler_dict"],
        full_band_wavelengths=PRODUCT_BANDS["tss"],
        use_diff=False,
        use_spectral_mask=False,
    )
    acdom_flat, acdom_2d, _, _, _ = preprocess_and_infer_emit_robust(
        nc_path=nc_path,
        model=models["acdom440"]["model"],
        scaler_Rrs=models["acdom440"]["scaler_Rrs"],
        TSS_scalers_dict=models["acdom440"]["scaler_dict"],
        full_band_wavelengths=PRODUCT_BANDS["acdom440"],
        use_diff=False,
        use_spectral_mask=False,
    )

    lat = np.ma.filled(np.asarray(lat), np.nan).astype(np.float64)
    lon = np.ma.filled(np.asarray(lon), np.nan).astype(np.float64)
    shape = lat.shape

    chla = chla_flat[:, 2].reshape(shape).astype(np.float32)
    tss = np.asarray(tss_2d, dtype=np.float32)
    acdom = np.asarray(acdom_2d, dtype=np.float32)

    return {
        "latitude": lat,
        "longitude": lon,
        "chla": chla,
        "tss": tss,
        "acdom440": acdom,
        "valid": int(np.isfinite(chla).sum()),
    }


def write_scene_cogs(maps, save_dir, date):
    """Write the in-memory product maps to date-named gridded COGs.

    Args:
        maps (dict): Output of :func:`infer_scene_maps`.
        save_dir (str): Output directory.
        date (str): Acquisition date (YYYYMMDD) used in the filename.

    Returns:
        list[str]: Paths to the written COGs.
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for var, label in PRODUCT_LABELS.items():
        paths.append(
            save_product_to_cog(
                out_tif=os.path.join(save_dir, f"S2-{date}-{label}.tif"),
                lat_2d=maps["latitude"],
                lon_2d=maps["longitude"],
                values_2d=maps[var],
            )
        )
    return paths


def save_products_to_nc(maps, output_nc):
    """Write the merged chl-a / TSS / aCDOM maps to a multi-variable NetCDF.

    The variables share the L2W swath geolocation (2D ``lat``/``lon``), so the
    output preserves the native scene geometry.

    Args:
        maps (dict): Output of :func:`infer_scene_maps`.
        output_nc (str): Output NetCDF path.

    Returns:
        str: The path to the written NetCDF.
    """
    import xarray as xr

    os.makedirs(os.path.dirname(output_nc) or ".", exist_ok=True)
    dims = ("y", "x")
    ds_out = xr.Dataset(
        data_vars={
            "chla": (
                dims,
                maps["chla"],
                {"long_name": "chlorophyll-a concentration", "units": "mg m-3"},
            ),
            "tss": (
                dims,
                maps["tss"],
                {"long_name": "total suspended solids", "units": "g m-3"},
            ),
            "acdom": (
                dims,
                maps["acdom440"],
                {
                    "long_name": "CDOM absorption coefficient at 440 nm",
                    "units": "m-1",
                },
            ),
            "lat": (
                dims,
                maps["latitude"],
                {"long_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                dims,
                maps["longitude"],
                {"long_name": "longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "title": "Sentinel-2 derived water quality products",
            "columns": "lat, lon, chla, tss, acdom",
        },
    )
    encoding = {
        "chla": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "tss": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "acdom": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "lat": {"zlib": True, "complevel": 4},
        "lon": {"zlib": True, "complevel": 4},
    }
    ds_out.to_netcdf(output_nc, encoding=encoding)
    ds_out.close()
    print("Saved NetCDF:", output_nc)
    return output_nc


def process_scene(
    safe_path,
    models,
    save_dir,
    l2_dir,
    acolite_dir=None,
    download=True,
    write_nc=True,
):
    """Run the full Sentinel-2 pipeline on one L1C ``.SAFE`` scene.

    Steps: ACOLITE atmospheric correction (L1C -> L2W) -> MoE-VAE inference ->
    write gridded Cloud Optimized GeoTIFFs (and optionally a merged NetCDF).

    Args:
        safe_path (str): Path to the Sentinel-2 ``.SAFE`` product directory.
        models (dict): Loaded models/scalers from :func:`load_models`.
        save_dir (str): Directory to write the COG/NetCDF products into.
        l2_dir (str): Root directory for intermediate ACOLITE L2 output.
        acolite_dir (str, optional): ACOLITE install directory (defaults to
            ``ACOLITE_DIR``; downloaded automatically if missing).
        download (bool): Download ACOLITE if not already installed (default
            True).
        write_nc (bool): Also write the merged products NetCDF (default True).

    Returns:
        list[str]: Paths to the written COG files.
    """
    print(f"Processing scene: {safe_path}")
    result = run_acolite(safe_path, l2_dir, acolite_dir=acolite_dir, download=download)
    l2w_path = result["l2w_files"][0]

    maps = infer_scene_maps(l2w_path, models)
    date = parse_acquisition_date(safe_path)
    cogs = write_scene_cogs(maps, save_dir, date)
    if write_nc:
        save_products_to_nc(maps, os.path.join(save_dir, f"S2-{date}-products.nc"))
    return cogs
