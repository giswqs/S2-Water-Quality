import os
import re
import subprocess
from pathlib import Path

import hypercoast
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import xarray as xr
from netCDF4 import Dataset
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def preprocess_pace_data_Robust(
    nc_path,
    scaler_Rrs,
    use_diff=False,
    full_band_wavelengths=None,
    use_spectral_mask=True,
):
    print(f"📥 Start processing: {nc_path}")

    try:
        PACE_dataset = hypercoast.read_pace(nc_path)
        print("✅ [1] Successfully read PACE data")

        da = PACE_dataset["Rrs"]
        Rrs = da.values  # [lat, lon, bands]
        latitude = da.latitude.values
        longitude = da.longitude.values
        print("✅ [2] Successfully retrieved Rrs, lat, and lon")

        # ============================
        # wavelength check
        # ============================
        if full_band_wavelengths is None:
            raise ValueError("full_band_wavelengths must be provided")

        if hasattr(da, "wavelength") or "wavelength" in da.coords:
            pace_band_wavelengths = da.wavelength.values
        else:
            raise ValueError("❌ Cannot extract wavelength")

        missing = [b for b in full_band_wavelengths if b not in pace_band_wavelengths]
        if missing:
            raise ValueError(f"❌ Missing wavelengths: {missing}")

        indices = [
            np.where(pace_band_wavelengths == b)[0][0] for b in full_band_wavelengths
        ]
        band_wavelengths = pace_band_wavelengths[indices]

        assert (
            band_wavelengths == np.array(full_band_wavelengths)
        ).all(), "❌ Band order mismatch"

        filtered_Rrs = Rrs[:, :, indices]
        print(f"✅ [3] Bands extracted: {len(indices)}")

        # ============================
        # 🔥 mask（核心改动）
        # ============================
        idx_440 = np.where(band_wavelengths == 440)[0][0]
        idx_560 = np.where(band_wavelengths == 560)[0][0]

        Rrs_440 = filtered_Rrs[:, :, idx_440]
        Rrs_560 = filtered_Rrs[:, :, idx_560]

        mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

        if use_spectral_mask:
            mask_condition = Rrs_560 >= Rrs_440
            mask = mask_nanfree & mask_condition
            print("✅ Spectral mask ENABLED")
        else:
            mask = mask_nanfree
            print("⚠️ Spectral mask DISABLED (only NaN)")

        print(f"✅ [4] Remaining pixels: {int(np.sum(mask))}")

        if not np.any(mask):
            raise ValueError("❌ No valid pixels passed filtering")

        valid_test_data = filtered_Rrs[mask]

        # ============================
        # smoothing + diff
        # ============================
        if use_diff:
            from scipy.ndimage import gaussian_filter1d

            Rrs_smoothed = np.array(
                [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
            )
            Rrs_processed = np.diff(Rrs_smoothed, axis=1)
            print("✅ [5] Gaussian smoothing + diff applied")
        else:
            Rrs_processed = valid_test_data
            print("✅ [5] No smoothing/diff")

        # ============================
        # normalization
        # ============================
        Rrs_normalized = scaler_Rrs.transform(
            torch.tensor(Rrs_processed, dtype=torch.float32)
        ).numpy()

        # ============================
        # DataLoader
        # ============================
        test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
        test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)

        print("✅ [6] DataLoader ready")

        return test_loader, filtered_Rrs, mask, latitude, longitude

    except Exception as e:
        print(f"❌ [ERROR] Failed to process file {nc_path}: {e}")
        return None


def infer_and_visualize_single_model_Robust(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    extent,
    rgb_image,
    structure_name,
    TSS_scalers_dict,
    vmin=0,
    vmax=50,
):
    device = next(model.parameters()).device
    predictions_all = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]

            # === Inverse transform using TSS_scalers_dict from training ===
            predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                torch.tensor(predictions.cpu().numpy(), dtype=torch.float32)
            )
            predictions_all.append(
                TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
            )

    predictions_all = np.vstack(predictions_all).squeeze(-1)
    # sanity check（非常重要）
    assert predictions_all.shape[0] == mask.sum()

    # 1. 空间回填 (H, W)
    outputs = np.full(mask.shape, np.nan, dtype=float)
    outputs[mask] = predictions_all

    # 2. 展平成 (N, 3)
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    chla_flat = outputs.flatten()

    final_output = np.column_stack((lat_flat, lon_flat, chla_flat))

    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)
    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    latitude_masked = final_output[:, 0]
    longitude_masked = final_output[:, 1]
    tss_values = final_output[:, 2]

    mean_lat = (extent[2] + extent[3]) / 2
    resolution_deg_lat = 1000 / 111000
    resolution_deg_lon = 1000 / (111000 * np.cos(np.radians(mean_lat)))
    grid_lon = np.arange(extent[0], extent[1], resolution_deg_lon)
    grid_lat = np.arange(extent[3], extent[2], -resolution_deg_lat)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    tss_resampled = griddata(
        (longitude_masked, latitude_masked),
        tss_values,
        (grid_lon, grid_lat),
        method="linear",
    )
    tss_resampled = np.ma.masked_invalid(tss_resampled)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")
    im = plt.imshow(
        tss_resampled,
        extent=extent,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    # cbar.set_label('(mg m$^{-3}$)', fontsize=16)
    plt.title(f"{structure_name}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

    return final_output


def preprocess_pace_data_minmax(
    nc_path,
    full_band_wavelengths=None,
    diff_before_norm=False,
    diff_after_norm=False,
    use_spectral_mask=True,  # 👈 新增
):
    try:
        # === Load data ===
        PACE_dataset = hypercoast.read_pace(nc_path)
        da = PACE_dataset["Rrs"]

        Rrs = da.values  # [lat, lon, bands]
        latitude = da.latitude.values
        longitude = da.longitude.values

        # ============================
        # Band check
        # ============================
        if full_band_wavelengths is None:
            raise ValueError("full_band_wavelengths must be provided")

        if hasattr(da, "wavelength") or "wavelength" in da.coords:
            pace_band_wavelengths = da.wavelength.values
        else:
            raise ValueError("❌ Cannot find wavelength info")

        missing = [b for b in full_band_wavelengths if b not in pace_band_wavelengths]
        if missing:
            raise ValueError(f"❌ Missing wavelengths: {missing}")

        indices = [
            np.where(pace_band_wavelengths == b)[0][0] for b in full_band_wavelengths
        ]
        band_wavelengths = pace_band_wavelengths[indices]

        assert (
            band_wavelengths == np.array(full_band_wavelengths)
        ).all(), "❌ Band order mismatch"

        filtered_Rrs = Rrs[:, :, indices]

        # ============================
        # 🔥 mask（核心改动）
        # ============================
        idx_440 = np.where(band_wavelengths == 440)[0][0]
        idx_560 = np.where(band_wavelengths == 560)[0][0]

        Rrs_440 = filtered_Rrs[:, :, idx_440]
        Rrs_560 = filtered_Rrs[:, :, idx_560]

        mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

        if use_spectral_mask:
            mask_condition = Rrs_560 >= Rrs_440
            mask = mask_nanfree & mask_condition
            print("✅ Spectral mask ENABLED")
        else:
            mask = mask_nanfree
            print("⚠️ Spectral mask DISABLED (only NaN)")

        print(f"Remaining pixels: {int(np.sum(mask))}")

        if not np.any(mask):
            raise ValueError("❌ No valid pixels passed the filtering.")

        valid_data = filtered_Rrs[mask]  # [N, B]

        # ============================
        # smoothing
        # ============================
        if diff_before_norm or diff_after_norm:
            from scipy.ndimage import gaussian_filter1d

            Rrs_smoothed = np.array(
                [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_data]
            )
            print("✅ Gaussian smoothing applied")
        else:
            Rrs_smoothed = valid_data
            print("✅ Smoothing not enabled")

        # ============================
        # diff before norm
        # ============================
        if diff_before_norm:
            Rrs_preprocessed = np.diff(Rrs_smoothed, axis=1)
            print("✅ Preprocessing before differencing completed")
        else:
            Rrs_preprocessed = Rrs_smoothed
            print("✅ Preprocessing before differencing not enabled")

        # ============================
        # normalization
        # ============================
        scalers = [MinMaxScaler((1, 10)) for _ in range(Rrs_preprocessed.shape[0])]

        Rrs_normalized = np.array(
            [
                scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
                for i, row in enumerate(Rrs_preprocessed)
            ]
        )

        # ============================
        # diff after norm
        # ============================
        if diff_after_norm:
            Rrs_normalized = np.diff(Rrs_normalized, axis=1)
            print("✅ Post-processing after differencing completed")
        else:
            print("✅ Post-processing after differencing not enabled")

        # ============================
        # DataLoader
        # ============================
        test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
        test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)

        return test_loader, Rrs, mask, latitude, longitude

    except Exception as e:
        print(f"❌ [ERROR] Failed to process file {nc_path}: {e}")
        return None


def preprocess_emit_data_Robust(
    nc_path,
    scaler_Rrs,
    use_diff=False,
    full_band_wavelengths=None,
    use_spectral_mask=True,
):

    if full_band_wavelengths is None:
        raise ValueError(
            "full_band_wavelengths must be provided to match EMIT Rrs bands"
        )

    def find_closest_band(target, available_bands):
        rrs_bands = [b for b in available_bands if b.startswith("Rrs_")]
        available_waves = [int(b.split("_")[1]) for b in rrs_bands]
        if not available_waves:
            raise ValueError("❌ No Rrs_* bands found in dataset")
        closest_wave = min(available_waves, key=lambda w: abs(w - target))
        return f"Rrs_{closest_wave}"

    dataset = Dataset(nc_path)

    latitude = dataset.variables["lat"][:]
    longitude = dataset.variables["lon"][:]

    all_vars = dataset.variables.keys()

    # ============================
    # band selection
    # ============================
    bands_to_extract = []
    for w in full_band_wavelengths:
        band_name = f"Rrs_{int(w)}"
        if band_name in all_vars:
            bands_to_extract.append(band_name)
        else:
            closest = find_closest_band(int(w), all_vars)
            print(f"⚠️ {band_name} does not exist, using {closest}")
            bands_to_extract.append(closest)

    filtered_Rrs = np.array([dataset.variables[band][:] for band in bands_to_extract])
    filtered_Rrs = np.moveaxis(filtered_Rrs, 0, -1)

    # ============================
    # 🔥 mask（核心改动）
    # ============================
    mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

    target_443 = (
        "Rrs_443"
        if "Rrs_443" in bands_to_extract
        else find_closest_band(443, bands_to_extract)
    )
    target_560 = (
        "Rrs_560"
        if "Rrs_560" in bands_to_extract
        else find_closest_band(560, bands_to_extract)
    )

    print(f"Using {target_443} and {target_560} for mask check.")

    idx_443 = bands_to_extract.index(target_443)
    idx_560 = bands_to_extract.index(target_560)

    if use_spectral_mask:
        mask_condition = filtered_Rrs[:, :, idx_443] <= filtered_Rrs[:, :, idx_560]
        mask = mask_nanfree & mask_condition
        print("✅ Spectral mask ENABLED")
    else:
        mask = mask_nanfree
        print("⚠️ Spectral mask DISABLED (only NaN)")

    print(f"Remaining pixels: {int(np.sum(mask))}")

    if not np.any(mask):
        raise ValueError("❌ No valid pixels")

    valid_test_data = filtered_Rrs[mask]

    # ============================
    # smooth + diff
    # ============================
    if use_diff:
        from scipy.ndimage import gaussian_filter1d

        Rrs_smoothed = np.array(
            [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
        )
        Rrs_processed = np.diff(Rrs_smoothed, axis=1)
        print("✅ [5] Performed Gaussian smoothing + first-order differencing")
    else:
        Rrs_processed = valid_test_data
        print("✅ [5] Smoothing and differencing not enabled")

    # ============================
    # normalize
    # ============================
    Rrs_normalized = scaler_Rrs.transform(
        torch.tensor(Rrs_processed, dtype=torch.float32)
    ).numpy()

    # ============================
    # DataLoader
    # ============================
    test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
    test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)

    print("✅ [6] DataLoader construction completed")

    return test_loader, filtered_Rrs, mask, latitude, longitude


def preprocess_emit_data_minmax(
    nc_path,
    full_band_wavelengths=None,
    diff_before_norm=False,
    diff_after_norm=False,
    use_spectral_mask=True,
):

    print(f"📥 Start processing: {nc_path}")

    if full_band_wavelengths is None or len(full_band_wavelengths) == 0:
        raise ValueError("full_band_wavelengths must be provided")

    full_band_wavelengths = [int(w) for w in full_band_wavelengths]

    try:
        with Dataset(nc_path) as dataset:

            latitude = dataset.variables["lat"][:]
            longitude = dataset.variables["lon"][:]

            all_vars = set(dataset.variables.keys())

            available_wavelengths = [
                float(v.split("_")[1]) for v in all_vars if v.startswith("Rrs_")
            ]

            def find_closest_band(target_nm):
                nearest = min(available_wavelengths, key=lambda w: abs(w - target_nm))
                return f"Rrs_{int(nearest)}"

            # ============================
            # band selection
            # ============================
            bands_to_extract = []
            for w in full_band_wavelengths:
                band_name = f"Rrs_{w}"
                if band_name in all_vars:
                    bands_to_extract.append(band_name)
                else:
                    closest = find_closest_band(w)
                    print(f"⚠️ {band_name} not found, using {closest}")
                    bands_to_extract.append(closest)

            seen = set()
            bands_to_extract = [
                b for b in bands_to_extract if not (b in seen or seen.add(b))
            ]

            if len(bands_to_extract) == 0:
                raise ValueError("❌ No usable bands")

            # ============================
            # stack
            # ============================
            Rrs_stack = []
            for band in bands_to_extract:
                Rrs_stack.append(dataset.variables[band][:])

            Rrs = np.array(Rrs_stack)
            Rrs = np.moveaxis(Rrs, 0, -1)
            filtered_Rrs = Rrs

            # ============================
            # find 440 & 560
            # ============================
            have_waves = [int(b.split("_")[1]) for b in bands_to_extract]

            def nearest_idx(target_nm):
                nearest_w = min(have_waves, key=lambda w: abs(w - target_nm))
                return bands_to_extract.index(f"Rrs_{nearest_w}")

            idx_440 = (
                bands_to_extract.index("Rrs_440")
                if "Rrs_440" in bands_to_extract
                else nearest_idx(440)
            )
            idx_560 = (
                bands_to_extract.index("Rrs_560")
                if "Rrs_560" in bands_to_extract
                else nearest_idx(560)
            )

            print(
                f"✅ Mask bands: {bands_to_extract[idx_440]} & {bands_to_extract[idx_560]}"
            )

            # ============================
            # 🔥 mask（核心）
            # ============================
            mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

            if use_spectral_mask:
                mask_condition = (
                    filtered_Rrs[:, :, idx_560] >= filtered_Rrs[:, :, idx_440]
                )
                mask = mask_nanfree & mask_condition
                print("✅ Spectral mask ENABLED")
            else:
                mask = mask_nanfree
                print("⚠️ Spectral mask DISABLED")

            print(f"Remaining pixels: {int(np.sum(mask))}")

            if not np.any(mask):
                raise ValueError("❌ No valid pixels")

            valid_test_data = filtered_Rrs[mask]

        # ============================
        # smoothing
        # ============================
        if diff_before_norm or diff_after_norm:
            from scipy.ndimage import gaussian_filter1d

            Rrs_smoothed = np.array(
                [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
            )
        else:
            Rrs_smoothed = valid_test_data

        # ============================
        # diff before
        # ============================
        if diff_before_norm:
            Rrs_preprocessed = np.diff(Rrs_smoothed, axis=1)
        else:
            Rrs_preprocessed = Rrs_smoothed

        # ============================
        # normalize
        # ============================
        scalers = [MinMaxScaler((1, 10)) for _ in range(Rrs_preprocessed.shape[0])]

        Rrs_normalized = np.array(
            [
                scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
                for i, row in enumerate(Rrs_preprocessed)
            ]
        )

        # ============================
        # diff after
        # ============================
        if diff_after_norm:
            Rrs_normalized = np.diff(Rrs_normalized, axis=1)

        # ============================
        # dataloader
        # ============================
        test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
        test_loader = DataLoader(test_tensor, batch_size=1024, shuffle=False)

        return test_loader, Rrs, mask, latitude, longitude

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def infer_and_visualize_single_model_minmax(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    extent,
    rgb_image,
    structure_name,
    vmin=0,
    vmax=50,
    log_offset=0.01,
):
    device = next(model.parameters()).device
    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]

            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset
            predictions_all.append(predictions_original)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # sanity check（非常重要）
    assert predictions_all.shape[0] == mask.sum()

    # 1. 空间回填 (H, W)
    outputs = np.full(mask.shape, np.nan, dtype=float)
    outputs[mask] = predictions_all

    # 2. 展平成 (N, 3)
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    chla_flat = outputs.flatten()

    final_output = np.column_stack((lat_flat, lon_flat, chla_flat))

    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)

    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")

    np.save(npy_path, final_output)

    latitude_masked = final_output[:, 0]
    longitude_masked = final_output[:, 1]
    tss_values = final_output[:, 2]

    mean_lat = (extent[2] + extent[3]) / 2
    resolution_deg_lat = 1000 / 111000
    resolution_deg_lon = 1000 / (111000 * np.cos(np.radians(mean_lat)))
    grid_lon = np.arange(extent[0], extent[1], resolution_deg_lon)
    grid_lat = np.arange(extent[3], extent[2], -resolution_deg_lat)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    from scipy.interpolate import griddata

    tss_resampled = griddata(
        (longitude_masked, latitude_masked),
        tss_values,
        (grid_lon, grid_lat),
        method="linear",
    )
    tss_resampled = np.ma.masked_invalid(tss_resampled)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")
    im = plt.imshow(
        tss_resampled,
        extent=extent,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("(mg m$^{-3}$)", fontsize=16)
    plt.title(f"{structure_name}", loc="left", fontsize=20)

    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

    return final_output


def infer_and_visualize_single_model_EMIT_Robust(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    rgb_nc_file,
    structure_name,
    TSS_scalers_dict,
    vmin=0,
    vmax=50,
    exposure_coefficient=5.0,
):
    device = next(model.parameters()).device
    predictions_all = []

    # === Model inference ===
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)

            # === Inverse transform using scalers ===
            predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                torch.tensor(output_dict["pred_y"].cpu().numpy(), dtype=torch.float32)
            )
            predictions_real = (
                TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
            )
            predictions_all.append(predictions_real)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # Fill predictions into 2D array according to mask
    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

    # Save as [lat, lon, value]
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))
    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)

    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    # === Construct RGB image from EMIT L2R .nc ===
    with Dataset(rgb_nc_file) as ds:
        # Latitude
        if "lat" in ds.variables:
            lat_var = ds.variables["lat"][:]
        elif "latitude" in ds.variables:
            lat_var = ds.variables["latitude"][:]
        else:
            raise KeyError("Latitude variable not found")

        # Longitude
        if "lon" in ds.variables:
            lon_var = ds.variables["lon"][:]
        elif "longitude" in ds.variables:
            lon_var = ds.variables["longitude"][:]
        else:
            raise KeyError("Longitude variable not found")

        # rhos band list
        band_list = []
        for name in ds.variables:
            m = re.match(r"^rhos_(\d+(?:\.\d+)?)$", name)
            if m:
                wl = float(m.group(1))
                band_list.append((wl, name))
        if not band_list:
            raise ValueError("No rhos_* bands found")

        # Select nearest RGB bands
        targets = {"R": 664.0, "G": 559.0, "B": 492.0}

        def pick_nearest(target_nm):
            return min(band_list, key=lambda x: abs(x[0] - target_nm))[1]

        var_R = pick_nearest(targets["R"])
        var_G = pick_nearest(targets["G"])
        var_B = pick_nearest(targets["B"])

        R = ds.variables[var_R][:]
        G = ds.variables[var_G][:]
        B = ds.variables[var_B][:]

        if isinstance(R, np.ma.MaskedArray):
            R = R.filled(np.nan)
        if isinstance(G, np.ma.MaskedArray):
            G = G.filled(np.nan)
        if isinstance(B, np.ma.MaskedArray):
            B = B.filled(np.nan)

    # Lat/lon grid
    if lat_var.ndim == 1 and lon_var.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat_var, lon_var, indexing="ij")
    else:
        lat2d, lon2d = lat_var, lon_var

    H, W = R.shape
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    R_flat, G_flat, B_flat = R.reshape(-1), G.reshape(-1), B.reshape(-1)

    lat_top, lat_bot = np.nanmax(lat2d), np.nanmin(lat2d)
    lon_min, lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
    grid_lat = np.linspace(lat_top, lat_bot, H)
    grid_lon = np.linspace(lon_min, lon_max, W)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    R_interp = griddata(
        (lon_flat, lat_flat), R_flat, (grid_lon, grid_lat), method="linear"
    )
    G_interp = griddata(
        (lon_flat, lat_flat), G_flat, (grid_lon, grid_lat), method="linear"
    )
    B_interp = griddata(
        (lon_flat, lat_flat), B_flat, (grid_lon, grid_lat), method="linear"
    )

    rgb_image = np.stack((R_interp, G_interp, B_interp), axis=-1)
    rgb_max = np.nanmax(rgb_image)
    if not np.isfinite(rgb_max) or rgb_max == 0:
        rgb_max = 1.0
    rgb_image = np.clip((rgb_image / rgb_max) * exposure_coefficient, 0, 1)
    extent_raw = [lon_min, lon_max, lat_bot, lat_top]

    # Interpolate predictions to same grid
    interp_output = griddata(
        (final_output[:, 1], final_output[:, 0]),  # lon, lat
        final_output[:, 2],
        (grid_lon, grid_lat),
        method="linear",
    )
    interp_output = np.ma.masked_invalid(interp_output)

    # Plot and save PNG
    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image, extent=extent_raw, origin="upper")
    im = plt.imshow(
        interp_output,
        extent=extent_raw,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    # cbar.set_label('(mg m$^{-3}$)', fontsize=16)
    plt.title(f"{structure_name}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {npy_path} (for npy_to_tif)")

    # Return numpy array for direct use
    return final_output


def infer_and_visualize_single_model_EMIT_minmax(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    rgb_nc_file,
    structure_name,
    vmin=0,
    vmax=50,
    log_offset=0.01,
    exposure_coefficient=5.0,
):
    device = next(model.parameters()).device
    predictions_all = []

    # === Model inference ===
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]
            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset
            predictions_all.append(predictions_original)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # Fill predictions into 2D array according to mask
    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

    # Flatten lat/lon and combine with predictions
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))
    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)

    # Save .npy file (lat, lon, value)
    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    # === Read RGB bands from .nc file ===
    with Dataset(rgb_nc_file) as ds:
        if "lat" in ds.variables:
            lat_var = ds.variables["lat"][:]
        elif "latitude" in ds.variables:
            lat_var = ds.variables["latitude"][:]
        else:
            raise KeyError("Latitude variable not found (lat/latitude)")

        if "lon" in ds.variables:
            lon_var = ds.variables["lon"][:]
        elif "longitude" in ds.variables:
            lon_var = ds.variables["longitude"][:]
        else:
            raise KeyError("Longitude variable not found (lon/longitude)")

        band_list = []
        for name in ds.variables.keys():
            m = re.match(r"^rhos_(\d+(?:\.\d+)?)$", name)
            if m:
                wl = float(m.group(1))
                band_list.append((wl, name))
        if not band_list:
            raise ValueError("No rhos_* bands found in file")

        targets = {"R": 664.0, "G": 559.0, "B": 492.0}

        def pick_nearest(target_nm):
            idx = int(np.argmin([abs(w - target_nm) for w, _ in band_list]))
            wl_sel, name_sel = band_list[idx]
            return wl_sel, name_sel

        wl_R, var_R = pick_nearest(targets["R"])
        wl_G, var_G = pick_nearest(targets["G"])
        wl_B, var_B = pick_nearest(targets["B"])

        print(
            f"RGB band selection: R→{var_R} (Δ{wl_R - targets['R']:+.1f}nm), "
            f"G→{var_G} (Δ{wl_G - targets['G']:+.1f}nm), "
            f"B→{var_B} (Δ{wl_B - targets['B']:+.1f}nm)"
        )

        R = ds.variables[var_R][:]
        G = ds.variables[var_G][:]
        B = ds.variables[var_B][:]
        if isinstance(R, np.ma.MaskedArray):
            R = R.filled(np.nan)
        if isinstance(G, np.ma.MaskedArray):
            G = G.filled(np.nan)
        if isinstance(B, np.ma.MaskedArray):
            B = B.filled(np.nan)

    if lat_var.ndim == 1 and lon_var.ndim == 1:
        lat2d, lon2d = np.meshgrid(
            np.asarray(lat_var), np.asarray(lon_var), indexing="ij"
        )
    else:
        lat2d, lon2d = np.asarray(lat_var), np.asarray(lon_var)

    H, W = R.shape
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    R_flat, G_flat, B_flat = R.reshape(-1), G.reshape(-1), B.reshape(-1)

    lat_top, lat_bot = np.nanmax(lat2d), np.nanmin(lat2d)
    lon_min, lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
    grid_lat = np.linspace(lat_top, lat_bot, H)
    grid_lon = np.linspace(lon_min, lon_max, W)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    R_interp = griddata(
        (lon_flat, lat_flat), R_flat, (grid_lon, grid_lat), method="linear"
    )
    G_interp = griddata(
        (lon_flat, lat_flat), G_flat, (grid_lon, grid_lat), method="linear"
    )
    B_interp = griddata(
        (lon_flat, lat_flat), B_flat, (grid_lon, grid_lat), method="linear"
    )

    rgb_image = np.stack((R_interp, G_interp, B_interp), axis=-1)
    rgb_max = np.nanmax(rgb_image)
    if not np.isfinite(rgb_max) or rgb_max == 0:
        rgb_max = 1.0
    rgb_image = np.clip((rgb_image / rgb_max) * exposure_coefficient, 0, 1)
    extent_raw = [lon_min, lon_max, lat_bot, lat_top]

    interp_output = griddata(
        (final_output[:, 1], final_output[:, 0]),
        final_output[:, 2],
        (grid_lon, grid_lat),
        method="linear",
    )
    interp_output = np.ma.masked_invalid(interp_output)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image, extent=extent_raw, origin="upper")
    im = plt.imshow(
        interp_output,
        extent=extent_raw,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    # cbar.set_label('(mg m$^{-3}$)', fontsize=16)
    plt.title(f"{structure_name}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {npy_path} (for npy_to_tif)")

    # Return numpy array for direct use
    return final_output


def preprocess_infer_pace_minmax(
    nc_path,
    model,
    full_band_wavelengths,
    use_spectral_mask=True,
    batch_size=2048,
    log_offset=1,
):
    import hypercoast
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, TensorDataset

    # ============================
    # Read PACE
    # ============================
    PACE_dataset = hypercoast.read_pace(nc_path)
    da = PACE_dataset["Rrs"]

    Rrs = da.values
    latitude = da.latitude.values
    longitude = da.longitude.values

    pace_band_wavelengths = da.wavelength.values

    indices = [
        np.where(pace_band_wavelengths == b)[0][0] for b in full_band_wavelengths
    ]

    filtered_Rrs = Rrs[:, :, indices]
    band_wavelengths = pace_band_wavelengths[indices]

    idx_440 = np.where(band_wavelengths == 440)[0][0]
    idx_560 = np.where(band_wavelengths == 560)[0][0]

    Rrs_440 = filtered_Rrs[:, :, idx_440]
    Rrs_560 = filtered_Rrs[:, :, idx_560]

    mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

    if use_spectral_mask:
        mask = mask_nanfree & (Rrs_560 >= Rrs_440)
        print("Spectral mask enabled")
    else:
        mask = mask_nanfree
        print("Spectral mask disabled")

    print(f"Remaining pixels: {int(mask.sum())}")

    if mask.sum() == 0:
        raise ValueError("No valid pixels")

    valid_data = filtered_Rrs[mask]

    scalers = [MinMaxScaler((1, 10)) for _ in range(valid_data.shape[0])]

    Rrs_normalized = np.array(
        [
            scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(valid_data)
        ]
    )

    # ============================
    # DataLoader
    # ============================
    test_loader = DataLoader(
        TensorDataset(torch.tensor(Rrs_normalized).float()),
        batch_size=batch_size,
        shuffle=False,
    )

    # ============================
    # Inference
    # ============================
    device = next(model.parameters()).device
    model.eval()
    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]

            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset
            predictions_all.append(predictions_original)

    predictions_all = np.vstack(predictions_all).squeeze(-1)
    assert predictions_all.shape[0] == mask.sum()

    # ============================
    # Spatial refill
    # ============================
    output_map = np.full(mask.shape, np.nan, dtype=np.float32)

    output_map[mask] = predictions_all

    # ============================
    # Convert to old 3-column format
    # ============================
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    pred_flat = output_map.flatten()

    final_output = np.column_stack((lat_flat, lon_flat, pred_flat))

    print(f"Output shape: {final_output.shape}")

    return final_output


def preprocess_infer_pace_robust(
    nc_path,
    model,
    scaler_Rrs,
    TSS_scalers_dict,
    full_band_wavelengths,
    use_diff=False,
    use_spectral_mask=True,
    batch_size=2048,
):
    import hypercoast
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        print(f"Start processing: {nc_path}")

        # ============================
        # Read PACE
        # ============================
        PACE_dataset = hypercoast.read_pace(nc_path)
        da = PACE_dataset["Rrs"]

        Rrs = da.values  # [lat, lon, bands]
        latitude = da.latitude.values
        longitude = da.longitude.values

        print("PACE data loaded")
        print("Rrs shape:", Rrs.shape)

        # ============================
        # Wavelength check
        # ============================
        if full_band_wavelengths is None:
            raise ValueError("full_band_wavelengths must be provided")

        if hasattr(da, "wavelength") or "wavelength" in da.coords:
            pace_band_wavelengths = da.wavelength.values
        else:
            raise ValueError("Cannot extract wavelength")

        missing = [b for b in full_band_wavelengths if b not in pace_band_wavelengths]

        if missing:
            raise ValueError(f"Missing wavelengths: {missing}")

        indices = [
            np.where(pace_band_wavelengths == b)[0][0] for b in full_band_wavelengths
        ]

        band_wavelengths = pace_band_wavelengths[indices]

        if not np.array_equal(band_wavelengths, np.array(full_band_wavelengths)):
            raise ValueError("Band order mismatch")

        filtered_Rrs = Rrs[:, :, indices]

        print(f"Bands extracted: {len(indices)}")
        print("Filtered Rrs shape:", filtered_Rrs.shape)

        # ============================
        # Spectral mask
        # ============================
        idx_440 = np.where(band_wavelengths == 440)[0][0]
        idx_560 = np.where(band_wavelengths == 560)[0][0]

        Rrs_440 = filtered_Rrs[:, :, idx_440]
        Rrs_560 = filtered_Rrs[:, :, idx_560]

        mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

        if use_spectral_mask:
            mask_condition = Rrs_560 >= Rrs_440
            mask = mask_nanfree & mask_condition
            print("Spectral mask enabled")
        else:
            mask = mask_nanfree
            print("Spectral mask disabled; only NaN filtering used")

        print(f"Remaining pixels: {int(mask.sum())}")

        if not np.any(mask):
            raise ValueError("No valid pixels passed filtering")

        valid_test_data = filtered_Rrs[mask]

        # ============================
        # Smoothing + diff
        # ============================
        if use_diff:
            from scipy.ndimage import gaussian_filter1d

            Rrs_smoothed = np.array(
                [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
            )

            Rrs_processed = np.diff(Rrs_smoothed, axis=1)

            print("Gaussian smoothing + diff applied")
        else:
            Rrs_processed = valid_test_data

            print("No smoothing/diff applied")

        # ============================
        # Input normalization
        # ============================
        Rrs_normalized = scaler_Rrs.transform(
            torch.tensor(Rrs_processed, dtype=torch.float32)
        ).numpy()

        print("Rrs normalized shape:", Rrs_normalized.shape)

        # ============================
        # DataLoader
        # ============================
        test_loader = DataLoader(
            TensorDataset(torch.tensor(Rrs_normalized).float()),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        # ============================
        # Inference
        # ============================
        model.eval()
        device = next(model.parameters()).device

        predictions_all = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch[0].to(device)

                output_dict = model(batch)
                predictions = output_dict["pred_y"]

                predictions_np = predictions.cpu().numpy()

                # Inverse transform using training scalers
                predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                    torch.tensor(predictions_np, dtype=torch.float32)
                )

                predictions_original = (
                    TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
                )

                predictions_all.append(predictions_original)

        predictions_all = np.vstack(predictions_all).squeeze(-1)

        if predictions_all.shape[0] != mask.sum():
            raise ValueError(
                f"Prediction mismatch: "
                f"{predictions_all.shape[0]} predictions vs "
                f"{mask.sum()} valid pixels"
            )

        print("Prediction shape:", predictions_all.shape)
        print(
            "Prediction min/max:",
            np.nanmin(predictions_all),
            np.nanmax(predictions_all),
        )

        # ============================
        # Spatial refill
        # ============================
        outputs = np.full(mask.shape, np.nan, dtype=np.float32)

        outputs[mask] = predictions_all

        # ============================
        # Convert to old 3-column format
        # ============================
        lat_flat = latitude.flatten()
        lon_flat = longitude.flatten()
        pred_flat = outputs.flatten()

        final_output = np.column_stack((lat_flat, lon_flat, pred_flat))

        if np.ma.isMaskedArray(final_output):
            final_output = final_output.filled(np.nan)

        return final_output

    except Exception as e:
        print(f"Failed to process {nc_path}: {e}")
        return None


def save_pace_products_to_nc(
    nc_path, save_dir, chla_output, tss_output, acdom_output, output_name=None
):
    import os

    import hypercoast
    import numpy as np

    # =====================================================
    # Read original PACE geometry
    # =====================================================
    PACE_dataset = hypercoast.read_pace(nc_path)

    da = PACE_dataset["Rrs"]

    latitude_2d = da.latitude.values
    longitude_2d = da.longitude.values

    shape = latitude_2d.shape

    print("Raster shape:", shape)

    # =====================================================
    # Extract variables
    # =====================================================
    chla = chla_output[:, 2]
    tss = tss_output[:, 2]
    acdom = acdom_output[:, 2]

    expected_pixels = shape[0] * shape[1]

    if len(chla) != expected_pixels:
        raise ValueError(f"Pixel mismatch: " f"{len(chla)} vs {expected_pixels}")

    # =====================================================
    # Reshape
    # =====================================================
    chla_map = chla.reshape(shape).astype(np.float32)

    tss_map = tss.reshape(shape).astype(np.float32)

    acdom_map = acdom.reshape(shape).astype(np.float32)

    # =====================================================
    # Output file
    # =====================================================
    if output_name is None:

        output_name = os.path.splitext(os.path.basename(nc_path))[0] + "_products.nc"

    output_nc = os.path.join(save_dir, output_name)

    # =====================================================
    # Create dataset
    # =====================================================
    out_ds = xr.Dataset(
        data_vars={
            "chla": (
                ("latitude", "longitude"),
                chla_map,
                {"long_name": "Chlorophyll-a", "units": "mg m-3"},
            ),
            "tss": (
                ("latitude", "longitude"),
                tss_map,
                {"long_name": "Total Suspended Solids", "units": "g m-3"},
            ),
            "acdom440": (
                ("latitude", "longitude"),
                acdom_map,
                {"long_name": "Absorption by CDOM at 440 nm", "units": "m-1"},
            ),
        },
        coords={
            "latitude": (
                ("latitude", "longitude"),
                latitude_2d,
                {"units": "degrees_north"},
            ),
            "longitude": (
                ("latitude", "longitude"),
                longitude_2d,
                {"units": "degrees_east"},
            ),
        },
        attrs={
            "title": "PACE Water Quality Products",
            "sensor": "PACE OCI",
            "source": "MoE-VAE Inference",
        },
    )

    # =====================================================
    # Compression
    # =====================================================
    encoding = {
        "chla": {"zlib": True, "complevel": 4},
        "tss": {"zlib": True, "complevel": 4},
        "acdom440": {"zlib": True, "complevel": 4},
    }

    # =====================================================
    # Save
    # =====================================================
    out_ds.to_netcdf(output_nc, encoding=encoding)

    print("Saved:", output_nc)

    return output_nc


def npy_to_tif(
    npy_input,
    out_tif,
    resolution_m=1000,
    method="linear",
    nodata_val=-9999.0,
    bbox_padding=0.0,
    lat_col=0,
    lon_col=1,
    band_cols=None,
    band_names=None,
    wavelengths=None,
    crs="EPSG:4326",
    compress="deflate",
    bigtiff="IF_SAFER",
):
    """
    Convert [lat, lon, band1, band2, ...] scattered points into a multi-band GeoTIFF (EPSG:4326).

    Parameters
    ----------
    - npy_input: str (path to .npy file) or ndarray (array of shape [N, M])
    - band_cols: which columns to rasterize as bands. Default: all columns after lat/lon.
    - band_names: optional list of band descriptions (len must match number of output bands).
    - wavelengths: optional list of wavelengths (e.g., [440, 619, 671]) to annotate descriptions.
    """

    # --- 1) Load data ---
    if isinstance(npy_input, str):
        arr = np.load(npy_input)
    elif isinstance(npy_input, np.ndarray):
        arr = npy_input
    else:
        raise TypeError("npy_input must be either a path string or a numpy.ndarray.")

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Input must be 2D with >=3 columns (lat, lon, values...).")

    lat = arr[:, lat_col].astype(float)
    lon = arr[:, lon_col].astype(float)

    # --- 2) Band selection ---
    if band_cols is None:
        band_cols = [i for i in range(arr.shape[1]) if i not in (lat_col, lon_col)]
    if isinstance(band_cols, (int, np.integer)):
        band_cols = [int(band_cols)]
    if len(band_cols) == 0:
        raise ValueError("No value columns selected for bands.")

    # --- 3) Bounds (+ padding) ---
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min -= bbox_padding
    lat_max += bbox_padding
    lon_min -= bbox_padding
    lon_max += bbox_padding

    # --- 4) Resolution conversion ---
    lat_center = (lat_min + lat_max) / 2.0
    deg_per_m_lat = 1.0 / 111000.0
    deg_per_m_lon = 1.0 / (111000.0 * np.cos(np.radians(lat_center)))
    res_lat_deg = resolution_m * deg_per_m_lat
    res_lon_deg = resolution_m * deg_per_m_lon

    # --- 5) Grid ---
    lon_axis = np.arange(lon_min, lon_max + res_lon_deg, res_lon_deg)
    lat_axis = np.arange(lat_min, lat_max + res_lat_deg, res_lat_deg)
    Lon, Lat = np.meshgrid(lon_axis, lat_axis)

    transform = from_origin(lon_axis.min(), lat_axis.max(), res_lon_deg, res_lat_deg)

    # --- 6) Interpolation ---
    grids = []
    for idx in band_cols:
        vals = arr[:, idx].astype(float)

        g = griddata(points=(lon, lat), values=vals, xi=(Lon, Lat), method=method)
        if np.isnan(g).any():
            g_near = griddata(
                points=(lon, lat), values=vals, xi=(Lon, Lat), method=method
            )
            g = np.where(np.isnan(g), g_near, g)

        grids.append(np.flipud(g).astype(np.float32))

    data_stack = np.stack(grids, axis=0)

    # --- 7) Write GeoTIFF ---
    profile = {
        "driver": "GTiff",
        "height": data_stack.shape[1],
        "width": data_stack.shape[2],
        "count": data_stack.shape[0],
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": bigtiff,
    }

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        for b in range(data_stack.shape[0]):
            band = data_stack[b]
            band[~np.isfinite(band)] = nodata_val
            dst.write(band, b + 1)

        # Descriptions
        n_bands = data_stack.shape[0]
        if band_names is not None and len(band_names) == n_bands:
            descriptions = list(map(str, band_names))
        elif wavelengths is not None and len(wavelengths) == n_bands:
            descriptions = [f"aphy_{int(wl)}" for wl in wavelengths]
        else:
            descriptions = [f"band_{band_cols[b]}" for b in range(n_bands)]

        for b in range(1, n_bands + 1):
            dst.set_band_description(b, descriptions[b - 1])

    print(f"✅ GeoTIFF saved: {out_tif}")


def preprocess_and_infer_emit_robust(
    nc_path,
    model,
    scaler_Rrs,
    TSS_scalers_dict,
    full_band_wavelengths,
    use_diff=False,
    use_spectral_mask=True,
    batch_size=2048,
):
    # =====================================================
    # Check inputs
    # =====================================================
    if full_band_wavelengths is None or len(full_band_wavelengths) == 0:
        raise ValueError("full_band_wavelengths must be provided")

    print(f"📥 Start processing: {nc_path}")

    # =====================================================
    # Helper: find closest Rrs band
    # =====================================================
    def find_closest_band(target, available_bands):
        rrs_bands = [b for b in available_bands if b.startswith("Rrs_")]
        available_waves = [int(b.split("_")[1]) for b in rrs_bands]

        if not available_waves:
            raise ValueError("❌ No Rrs_* bands found in dataset")

        closest_wave = min(available_waves, key=lambda w: abs(w - target))

        return f"Rrs_{closest_wave}"

    # =====================================================
    # Read NC
    # =====================================================
    with Dataset(nc_path) as dataset:

        latitude = dataset.variables["lat"][:]
        longitude = dataset.variables["lon"][:]

        all_vars = list(dataset.variables.keys())

        # =================================================
        # Band selection
        # =================================================
        bands_to_extract = []

        for w in full_band_wavelengths:
            band_name = f"Rrs_{int(w)}"

            if band_name in all_vars:
                bands_to_extract.append(band_name)
            else:
                closest = find_closest_band(int(w), all_vars)
                print(f"⚠️ {band_name} does not exist, using {closest}")
                bands_to_extract.append(closest)

        # Remove duplicated bands
        seen = set()
        bands_to_extract = [
            b for b in bands_to_extract if not (b in seen or seen.add(b))
        ]

        if len(bands_to_extract) == 0:
            raise ValueError("❌ No usable Rrs bands")

        print(f"✅ Using {len(bands_to_extract)} bands")

        # =================================================
        # Stack Rrs
        # =================================================
        Rrs_stack = []

        for band in bands_to_extract:
            arr = dataset.variables[band][:]

            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)

            Rrs_stack.append(arr)

        filtered_Rrs = np.array(Rrs_stack)
        filtered_Rrs = np.moveaxis(filtered_Rrs, 0, -1)

        # =================================================
        # Mask
        # =================================================
        mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)

        target_443 = (
            "Rrs_443"
            if "Rrs_443" in bands_to_extract
            else find_closest_band(443, bands_to_extract)
        )

        target_560 = (
            "Rrs_560"
            if "Rrs_560" in bands_to_extract
            else find_closest_band(560, bands_to_extract)
        )

        print(f"✅ Mask bands: {target_443} & {target_560}")

        idx_443 = bands_to_extract.index(target_443)
        idx_560 = bands_to_extract.index(target_560)

        if use_spectral_mask:
            mask_condition = filtered_Rrs[:, :, idx_443] <= filtered_Rrs[:, :, idx_560]
            mask = mask_nanfree & mask_condition
            print("✅ Spectral mask ENABLED")
        else:
            mask = mask_nanfree
            print("⚠️ Spectral mask DISABLED")

        print(f"Remaining pixels: {int(np.sum(mask))}")

        if not np.any(mask):
            raise ValueError("❌ No valid pixels")

        valid_test_data = filtered_Rrs[mask]

    # =====================================================
    # Smooth + diff
    # =====================================================
    if use_diff:
        from scipy.ndimage import gaussian_filter1d

        Rrs_smoothed = np.array(
            [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
        )

        Rrs_processed = np.diff(Rrs_smoothed, axis=1)

        print("✅ Gaussian smoothing + first-order differencing enabled")
    else:
        Rrs_processed = valid_test_data
        print("⚠️ Smoothing and differencing disabled")

    # =====================================================
    # Normalize with Robust scaler
    # =====================================================
    Rrs_normalized = scaler_Rrs.transform(
        torch.tensor(Rrs_processed, dtype=torch.float32)
    ).numpy()

    # =====================================================
    # DataLoader
    # =====================================================
    test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())

    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    print("✅ DataLoader construction completed")

    # =====================================================
    # Model inference
    # =====================================================
    device = next(model.parameters()).device
    model.eval()

    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)

            output_dict = model(batch)

            predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                torch.tensor(output_dict["pred_y"].cpu().numpy(), dtype=torch.float32)
            )

            predictions_real = (
                TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
            )

            predictions_all.append(predictions_real)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # =====================================================
    # Fill predictions back to 2D image
    # =====================================================
    outputs = np.full((filtered_Rrs.shape[0], filtered_Rrs.shape[1]), np.nan)

    outputs[mask] = predictions_all

    # =====================================================
    # Save as [lat, lon, value]
    # =====================================================
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()

    final_output = np.column_stack([lat_flat, lon_flat, output_flat])

    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)

    return final_output, outputs, mask, latitude, longitude


def preprocess_and_infer_emit_minmax(
    nc_path,
    model,
    full_band_wavelengths,
    diff_before_norm=False,
    diff_after_norm=False,
    use_spectral_mask=True,
    batch_size=1024,
    log_offset=1,
):
    print(f"📥 Start processing: {nc_path}")

    if full_band_wavelengths is None or len(full_band_wavelengths) == 0:
        raise ValueError("full_band_wavelengths must be provided")

    full_band_wavelengths = [int(w) for w in full_band_wavelengths]

    # =====================================================
    # Read NC and preprocess
    # =====================================================
    with Dataset(nc_path) as dataset:

        latitude = dataset.variables["lat"][:]
        longitude = dataset.variables["lon"][:]

        all_vars = set(dataset.variables.keys())

        available_wavelengths = [
            float(v.split("_")[1]) for v in all_vars if v.startswith("Rrs_")
        ]

        if len(available_wavelengths) == 0:
            raise ValueError("❌ No Rrs_* bands found")

        def find_closest_band(target_nm):
            nearest = min(available_wavelengths, key=lambda w: abs(w - target_nm))
            return f"Rrs_{int(nearest)}"

        # ----------------------------
        # Band selection
        # ----------------------------
        bands_to_extract = []

        for w in full_band_wavelengths:
            band_name = f"Rrs_{w}"

            if band_name in all_vars:
                bands_to_extract.append(band_name)
            else:
                closest = find_closest_band(w)
                print(f"⚠️ {band_name} not found, using {closest}")
                bands_to_extract.append(closest)

        # Remove duplicated bands
        seen = set()
        bands_to_extract = [
            b for b in bands_to_extract if not (b in seen or seen.add(b))
        ]

        if len(bands_to_extract) == 0:
            raise ValueError("❌ No usable bands")

        print(f"✅ Using {len(bands_to_extract)} bands")

        # ----------------------------
        # Stack Rrs
        # ----------------------------
        Rrs_stack = []

        for band in bands_to_extract:
            arr = dataset.variables[band][:]

            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)

            Rrs_stack.append(arr)

        Rrs = np.array(Rrs_stack)
        Rrs = np.moveaxis(Rrs, 0, -1)

        # ----------------------------
        # Find 440 and 560 bands
        # ----------------------------
        have_waves = [int(b.split("_")[1]) for b in bands_to_extract]

        def nearest_idx(target_nm):
            nearest_w = min(have_waves, key=lambda w: abs(w - target_nm))
            return bands_to_extract.index(f"Rrs_{nearest_w}")

        idx_440 = (
            bands_to_extract.index("Rrs_440")
            if "Rrs_440" in bands_to_extract
            else nearest_idx(440)
        )

        idx_560 = (
            bands_to_extract.index("Rrs_560")
            if "Rrs_560" in bands_to_extract
            else nearest_idx(560)
        )

        print(
            f"✅ Mask bands: "
            f"{bands_to_extract[idx_440]} & {bands_to_extract[idx_560]}"
        )

        # ----------------------------
        # Mask
        # ----------------------------
        mask_nanfree = np.all(~np.isnan(Rrs), axis=2)

        if use_spectral_mask:
            mask_condition = Rrs[:, :, idx_560] >= Rrs[:, :, idx_440]
            mask = mask_nanfree & mask_condition
            print("✅ Spectral mask ENABLED")
        else:
            mask = mask_nanfree
            print("⚠️ Spectral mask DISABLED")

        print(f"Remaining pixels: {int(np.sum(mask))}")

        if not np.any(mask):
            raise ValueError("❌ No valid pixels")

        valid_test_data = Rrs[mask]

    # =====================================================
    # Optional smoothing
    # =====================================================
    if diff_before_norm or diff_after_norm:
        from scipy.ndimage import gaussian_filter1d

        Rrs_smoothed = np.array(
            [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
        )
    else:
        Rrs_smoothed = valid_test_data

    # =====================================================
    # Diff before normalization
    # =====================================================
    if diff_before_norm:
        Rrs_preprocessed = np.diff(Rrs_smoothed, axis=1)
    else:
        Rrs_preprocessed = Rrs_smoothed

    # =====================================================
    # Row-wise MinMax normalization
    # =====================================================
    Rrs_normalized = np.array(
        [
            MinMaxScaler((1, 10)).fit_transform(row.reshape(-1, 1)).flatten()
            for row in Rrs_preprocessed
        ]
    )

    # =====================================================
    # Diff after normalization
    # =====================================================
    if diff_after_norm:
        Rrs_normalized = np.diff(Rrs_normalized, axis=1)

    # =====================================================
    # Dataloader
    # =====================================================
    test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    # =====================================================
    # Model inference
    # =====================================================
    device = next(model.parameters()).device
    model.eval()

    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)

            output_dict = model(batch)
            predictions = output_dict["pred_y"]

            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset

            predictions_all.append(predictions_original)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # =====================================================
    # Fill prediction back to 2D grid
    # =====================================================
    output_map = np.full(mask.shape, np.nan, dtype=np.float32)

    output_map[mask] = predictions_all

    # ============================
    # Convert to old 3-column format
    # ============================
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    pred_flat = output_map.flatten()

    final_output = np.column_stack((lat_flat, lon_flat, pred_flat))

    print(f"Output shape: {final_output.shape}")

    return final_output


def run_emit_acolite_single(
    input_nc,
    out_root,
    acolite="/home/jiangli/acolite_py_linux_2025/acolite",
    proj_dir="/home/jiangli/acolite_py_linux_2025/dist/acolite/_internal/share/proj",
):
    """
    Process one specified EMIT RAD .nc file with ACOLITE.

    Returns
    -------
    result : dict
        {
            "scene": scene_name,
            "input_nc": input_nc,
            "output_dir": output_dir,
            "l2r_files": list of L2R nc paths,
            "l2w_files": list of L2W nc paths
        }
    """

    input_nc = Path(input_nc)
    out_root = Path(out_root)

    if not input_nc.exists():
        raise FileNotFoundError(f"Input NC does not exist: {input_nc}")

    if input_nc.suffix != ".nc":
        raise ValueError(f"Input file must be .nc: {input_nc}")

    out_root.mkdir(parents=True, exist_ok=True)

    scene_name = input_nc.stem
    out_path = out_root / scene_name
    out_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PROJ_LIB"] = proj_dir
    env["PROJ_DATA"] = proj_dir

    template = """
polygon=None
limit=None

# ===== Basic =====
atmospheric_correction=True

# ===== Output =====
l2w_parameters=Rrs_*

rgb_rhot=True
rgb_rhos=True
map_l2w=True

# ===== Water mask =====
l2w_mask=True
l2w_mask_water_parameters=True

# ===== EMIT water mask wave =====
l2w_mask_wave=790
l2w_mask_threshold=0.08

# ===== Cirrus mask =====
l2w_mask_cirrus=True
l2w_mask_cirrus_wave=1014
l2w_mask_cirrus_threshold=0.12

# ===== High TOA mask =====
l2w_mask_high_toa=True
l2w_mask_high_toa_threshold=10

# ===== Negative rhow mask =====
l2w_mask_negative_rhow=True
l2w_mask_negative_wave_range=410,730

# ===== Smooth mask =====
l2w_mask_smooth=True
"""

    settings = f"""
inputfile={input_nc}
output={out_path}
{template}
"""

    settings_file = Path("/tmp") / f"{scene_name}_acolite_settings.txt"

    with open(settings_file, "w") as f:
        f.write(settings)

    cmd = [str(acolite), "--cli", "--settings", str(settings_file)]

    print("=" * 80)
    print("Running ACOLITE for EMIT RAD:")
    print(input_nc)
    print("=" * 80)

    subprocess.run(cmd, env=env, check=True)

    nc_outputs = sorted(out_path.glob("*.nc"))

    l2r_files = [str(p) for p in nc_outputs if "L2R" in p.name.upper()]

    l2w_files = [str(p) for p in nc_outputs if "L2W" in p.name.upper()]

    if not l2w_files:
        raise FileNotFoundError(f"No L2W file generated in: {out_path}")

    result = {
        "scene": scene_name,
        "input_nc": str(input_nc),
        "output_dir": str(out_path),
        "l2r_files": l2r_files,
        "l2w_files": l2w_files,
    }

    print("\nACOLITE finished.")
    print("Output directory:")
    print(out_path)

    print("\nL2R files:")
    for p in l2r_files:
        print("  ", p)

    print("\nL2W files:")
    for p in l2w_files:
        print("  ", p)

    return result


def run_acolite_safe(
    safe_path,
    out_root,
    acolite="/home/jiangli/acolite_py_linux_2025/acolite",
    proj_dir="/home/jiangli/acolite_py_linux_2025/dist/acolite/_internal/share/proj",
):
    """
    Process a single Sentinel-2 SAFE with ACOLITE.

    Parameters
    ----------
    safe_path : str
        Path to *.SAFE
    out_root : str
        Root output directory

    Returns
    -------
    dict
        {
            "scene": scene_name,
            "output_dir": output_dir,
            "l2r_files": [...],
            "l2w_files": [...]
        }
    """

    safe_path = Path(safe_path)

    if not safe_path.exists():
        raise FileNotFoundError(safe_path)

    scene_name = safe_path.stem

    out_path = Path(out_root) / scene_name
    out_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PROJ_LIB"] = proj_dir
    env["PROJ_DATA"] = proj_dir

    template = """
polygon=None
limit=None

atmospheric_correction=True
atmospheric_correction_method=radcor

l2w_parameters=Rrs_*

rgb_rhot=True
rgb_rhos=True
map_l2w=True

l2w_mask_wave=1600
l2w_mask_threshold=0.1
"""

    settings = f"""
inputfile={safe_path}
output={out_path}
{template}
"""

    settings_file = Path("/tmp") / f"{scene_name}_acolite_settings.txt"

    with open(settings_file, "w") as f:
        f.write(settings)

    cmd = [acolite, "--cli", "--settings", str(settings_file)]

    print(f"Running ACOLITE: {scene_name}")

    subprocess.run(cmd, env=env, check=True)

    nc_files = sorted(out_path.glob("*.nc"))

    l2r_files = [str(f) for f in nc_files if "L2R" in f.name.upper()]

    l2w_files = [str(f) for f in nc_files if "L2W" in f.name.upper()]

    result = {
        "scene": scene_name,
        "output_dir": str(out_path),
        "l2r_files": l2r_files,
        "l2w_files": l2w_files,
    }

    return result
