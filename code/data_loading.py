import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.utils.data import Subset
from preprocess import RobustMinMaxScaler, LogScaler


def load_real_data_Robust(
    excel_path,
    selected_bands,
    target_parameter="TSS",
    split_ratio=0.7,
    seed=42,
    use_diff=False,
    lower_quantile=0.0,
    upper_quantile=1.0,
    Rrs_range=(0, 0.25),
    target_range=(-0.5, 0.5),
):

    rounded_bands = [int(round(b)) for b in selected_bands]
    band_cols = [f"Rrs_{b}" for b in rounded_bands]

    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    df_rrs_selected = df_rrs[["GLORIA_ID"] + band_cols]
    df_param_selected = df_param[["GLORIA_ID", target_parameter]]
    df_merged = pd.merge(
        df_rrs_selected, df_param_selected, on="GLORIA_ID", how="inner"
    )

    mask_rrs_valid = df_merged[band_cols].notna().all(axis=1)
    mask_param_valid = df_merged[target_parameter].notna()
    df_filtered = df_merged[mask_rrs_valid & mask_param_valid].reset_index(drop=True)

    print(
        f"Number of samples after filtering Rrs and {target_parameter}: {len(df_filtered)}"
    )

    lower = df_filtered[target_parameter].quantile(lower_quantile)
    top = df_filtered[target_parameter].quantile(upper_quantile)
    df_filtered = df_filtered[
        (df_filtered[target_parameter] >= lower)
        & (df_filtered[target_parameter] <= top)
    ].reset_index(drop=True)

    print(
        f"Number of samples after removing {target_parameter} quantiles [{lower_quantile}, {upper_quantile}]: {len(df_filtered)}"
    )

    all_sample_ids = df_filtered["GLORIA_ID"].astype(str).tolist()
    Rrs_array = df_filtered[band_cols].values
    param_array = df_filtered[[target_parameter]].values

    if use_diff:
        Rrs_array = np.diff(Rrs_array, axis=1)

    scaler_Rrs = RobustMinMaxScaler(feature_range=Rrs_range)
    scaler_Rrs.fit(torch.tensor(Rrs_array, dtype=torch.float32))
    Rrs_normalized = scaler_Rrs.transform(
        torch.tensor(Rrs_array, dtype=torch.float32)
    ).numpy()

    log_scaler = LogScaler(shift_min=False, safety_term=1e-8)
    param_log = log_scaler.fit_transform(torch.tensor(param_array, dtype=torch.float32))
    param_scaler = RobustMinMaxScaler(
        feature_range=target_range, global_scale=True, robust=True
    )
    param_transformed = param_scaler.fit_transform(param_log).numpy()

    Rrs_tensor = torch.tensor(Rrs_normalized, dtype=torch.float32)
    param_tensor = torch.tensor(param_transformed, dtype=torch.float32)
    dataset = TensorDataset(Rrs_tensor, param_tensor)

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_size = int(split_ratio * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_ids = [all_sample_ids[i] for i in train_indices]
    test_ids = [all_sample_ids[i] for i in test_indices]

    train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = param_tensor.shape[1]
    TSS_scalers_dict = {"log": log_scaler, "robust": param_scaler}

    return (
        train_dl,
        test_dl,
        input_dim,
        output_dim,
        train_ids,
        test_ids,
        scaler_Rrs,
        TSS_scalers_dict,
    )


def load_real_test_Robust(
    excel_path,
    selected_bands,
    max_allowed_diff=1.0,
    scaler_Rrs=None,
    scalers_dict=None,
    use_diff=False,
    target_parameter="SPM",
):

    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if df_rrs.shape[0] != df_param.shape[0]:
        raise ValueError(
            f"❌ The number of rows in the Rrs table and parameter table do not match. Rrs: {df_rrs.shape[0]}, parameter: {df_param.shape[0]}"
        )

    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = df_rrs["Date"].astype(str).tolist()

    # Match target bands
    rrs_wavelengths = []
    rrs_cols = []
    for col in df_rrs.columns:
        try:
            wl = float(col)
            rrs_wavelengths.append(wl)
            rrs_cols.append(col)
        except:
            continue

    band_cols = []
    for target_band in selected_bands:
        diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
        min_diff = min(diffs)
        if min_diff > max_allowed_diff:
            raise ValueError(
                f"Target wavelength {target_band} nm cannot be matched, error {min_diff:.2f} nm exceeds the allowed range"
            )
        best_idx = diffs.index(min_diff)
        band_cols.append(rrs_cols[best_idx])

    print(f"\n✅ Band matching successful, {len(selected_bands)} target bands in total")
    print(f"Final number of valid test samples: {df_rrs.shape[0]}\n")

    Rrs_array = df_rrs[band_cols].values
    param_array = df_param[[target_parameter]].values.flatten()
    # === Key: Remove rows with NaN/Inf before differencing ===
    mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_target_ok = np.isfinite(param_array)
    mask_ok = mask_inputs_ok & mask_target_ok
    if not np.any(mask_ok):
        raise ValueError("❌ Valid samples = 0 (NaN/Inf found in input or target).")
    dropped = int(len(mask_ok) - mask_ok.sum())
    if dropped > 0:
        print(
            f"⚠️ Dropped {dropped} invalid samples (containing NaN/Inf) before differencing"
        )

    Rrs_array = Rrs_array[mask_ok]
    param_array = param_array[mask_ok]
    sample_ids = [sid for sid, keep in zip(sample_ids, mask_ok) if keep]
    sample_dates = [d for d, keep in zip(sample_dates, mask_ok) if keep]

    if use_diff:
        Rrs_array = np.diff(Rrs_array, axis=1)

    Rrs_tensor = torch.tensor(Rrs_array, dtype=torch.float32)
    Rrs_normalized = scaler_Rrs.transform(Rrs_tensor).numpy()

    log_scaler = scalers_dict["log"]
    robust_scaler = scalers_dict["robust"]
    param_log = log_scaler.transform(
        torch.tensor(param_array.reshape(-1, 1), dtype=torch.float32)
    )
    param_transformed = robust_scaler.transform(param_log).numpy()

    dataset = TensorDataset(
        torch.tensor(Rrs_normalized, dtype=torch.float32),
        torch.tensor(param_transformed.reshape(-1, 1), dtype=torch.float32),
    )
    test_dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = 1

    return test_dl, input_dim, output_dim, sample_ids, sample_dates


def build_real_train_test_by_date_robust(
    excel_path,
    selected_bands,
    scaler_Rrs,
    scalers_dict,
    target_parameter="SPM",
    seed=42,
    max_allowed_diff=1.0,
    use_diff=False,
    train_batch_size=1024,
):

    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # ================= READ =================
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if len(df_rrs) != len(df_param):
        raise ValueError("Rrs and parameter row mismatch")

    df_rrs["Date"] = pd.to_datetime(df_rrs["Date"])
    df_param["Date"] = pd.to_datetime(df_param["Date"])

    # ================= SPLIT HALF WITHIN EACH DATE =================
    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    for date, group in df_rrs.groupby("Date"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        half = len(indices) // 2

        train_indices.extend(indices[:half])
        test_indices.extend(indices[half:])

    # build train/test
    df_rrs_train = df_rrs.loc[train_indices].reset_index(drop=True)
    df_param_train = df_param.loc[train_indices].reset_index(drop=True)

    df_rrs_test = df_rrs.loc[test_indices].reset_index(drop=True)
    df_param_test = df_param.loc[test_indices].reset_index(drop=True)

    # ================= INNER BUILDER =================
    def build_loader(
        df_rrs_local, df_param_local, shuffle, batch_size, return_meta=False
    ):

        sample_ids = df_rrs_local["Site Label"].astype(str).tolist()
        sample_dates = df_rrs_local["Date"].astype(str).tolist()
        lats = df_rrs_local["Lat"].astype(float).tolist()
        lons = df_rrs_local["Long"].astype(float).tolist()
        areas = df_param_local["Area"].astype(str).tolist()

        # ---- wavelength match ----
        rrs_wavelengths = []
        rrs_cols = []
        for col in df_rrs_local.columns:
            try:
                wl = float(col)
                rrs_wavelengths.append(wl)
                rrs_cols.append(col)
            except:
                continue

        band_cols = []
        for target_band in selected_bands:
            diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
            min_diff = min(diffs)
            if min_diff > max_allowed_diff:
                raise ValueError(f"Band {target_band} unmatched ({min_diff:.2f} nm)")
            band_cols.append(rrs_cols[diffs.index(min_diff)])

        # ---- arrays ----
        Rrs_array = df_rrs_local[band_cols].values.astype(float)
        target_array = df_param_local[[target_parameter]].values.astype(float).flatten()

        # ---- remove NaN ----
        mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
        mask_target_ok = np.isfinite(target_array)
        mask_ok = mask_inputs_ok & mask_target_ok

        Rrs_array = Rrs_array[mask_ok]
        target_array = target_array[mask_ok]

        sample_ids_f = [x for x, m in zip(sample_ids, mask_ok) if m]
        sample_dates_f = [x for x, m in zip(sample_dates, mask_ok) if m]
        lats_f = [x for x, m in zip(lats, mask_ok) if m]
        lons_f = [x for x, m in zip(lons, mask_ok) if m]
        areas_f = [x for x, m in zip(areas, mask_ok) if m]

        # ---- diff ----
        if use_diff:
            Rrs_array = np.diff(Rrs_array, axis=1)

        # ---- Rrs robust normalization ----
        Rrs_tensor = torch.tensor(Rrs_array, dtype=torch.float32)
        Rrs_norm = scaler_Rrs.transform(Rrs_tensor).numpy()

        # ---- target robust transform ----
        log_scaler = scalers_dict["log"]
        robust_scaler = scalers_dict["robust"]

        param_log = log_scaler.transform(
            torch.tensor(target_array.reshape(-1, 1), dtype=torch.float32)
        )
        target_transformed = robust_scaler.transform(param_log).numpy()

        # ---- tensors ----
        X = torch.tensor(Rrs_norm, dtype=torch.float32)
        y = torch.tensor(target_transformed.reshape(-1, 1), dtype=torch.float32)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if return_meta:
            meta = {
                "sample_ids": sample_ids_f,
                "dates": sample_dates_f,
                "lat": lats_f,
                "lon": lons_f,
                "area": areas_f,
            }
            return loader, meta
        else:
            return loader

    # ================= BUILD =================
    train_loader = build_loader(df_rrs_train, df_param_train, True, train_batch_size)
    test_loader, test_meta = build_loader(
        df_rrs_test, df_param_test, False, len(df_rrs_test), True
    )

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Test samples : {len(test_loader.dataset)}")

    return train_loader, test_loader, test_meta


def load_real_data(
    excel_path,
    selected_bands,
    split_ratio=0.7,
    seed=42,
    diff_before_norm=False,
    diff_after_norm=False,
    target_parameter="TSS",
    lower_quantile=0.0,
    upper_quantile=1.0,
    log_offset=0.01,
):

    rounded_bands = [int(round(b)) for b in selected_bands]
    band_cols = [f"Rrs_{b}" for b in rounded_bands]
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")
    df_rrs_selected = df_rrs[["GLORIA_ID"] + band_cols]
    df_param_selected = df_param[["GLORIA_ID", target_parameter]]
    df_merged = pd.merge(
        df_rrs_selected, df_param_selected, on="GLORIA_ID", how="inner"
    )

    # === Filter valid samples ===
    mask_rrs_valid = df_merged[band_cols].notna().all(axis=1)
    mask_target_valid = df_merged[target_parameter].notna()
    df_filtered = df_merged[mask_rrs_valid & mask_target_valid].reset_index(drop=True)
    print(
        f"✅ Number of samples after filtering Rrs and {target_parameter}: {len(df_filtered)}"
    )

    # === Quantile clipping for target parameter ===
    lower = df_filtered[target_parameter].quantile(lower_quantile)
    upper = df_filtered[target_parameter].quantile(upper_quantile)
    df_filtered = df_filtered[
        (df_filtered[target_parameter] >= lower)
        & (df_filtered[target_parameter] <= upper)
    ].reset_index(drop=True)
    print(
        f"✅ Number of samples after removing {target_parameter} quantiles [{lower_quantile}, {upper_quantile}]: {len(df_filtered)}"
    )

    # === Extract sample IDs, Rrs, and target parameter ===
    all_sample_ids = df_filtered["GLORIA_ID"].astype(str).tolist()
    Rrs_array = df_filtered[band_cols].values
    param_array = df_filtered[[target_parameter]].values

    if diff_before_norm:
        Rrs_array = np.diff(Rrs_array, axis=1)

    # === Apply MinMax scaling to [1, 10] for each sample independently ===
    scalers_Rrs_real = [MinMaxScaler((1, 10)) for _ in range(Rrs_array.shape[0])]
    Rrs_normalized = np.array(
        [
            scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    if diff_after_norm:
        Rrs_normalized = np.diff(Rrs_normalized, axis=1)

    # === Transform target parameter to log10(param + log_offset) ===
    param_transformed = np.log10(param_array + log_offset)

    # === Build Dataset ===
    Rrs_tensor = torch.tensor(Rrs_normalized, dtype=torch.float32)
    param_tensor = torch.tensor(param_transformed, dtype=torch.float32)
    dataset = TensorDataset(Rrs_tensor, param_tensor)

    # === Split into training and testing sets ===
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_size = int(split_ratio * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_ids = [all_sample_ids[i] for i in train_indices]
    test_ids = [all_sample_ids[i] for i in test_indices]

    train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = param_tensor.shape[1]

    return (train_dl, test_dl, input_dim, output_dim, train_ids, test_ids)


def load_real_test(
    excel_path,
    selected_bands,
    max_allowed_diff=1.0,
    diff_before_norm=False,
    diff_after_norm=False,
    target_parameter="TSS",
    log_offset=0.01,
):

    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if df_rrs.shape[0] != df_param.shape[0]:
        raise ValueError(
            f"❌ The number of rows in the Rrs table and parameter table do not match. Rrs: {df_rrs.shape[0]}, parameter: {df_param.shape[0]}"
        )

    # === Extract IDs and dates ===
    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = df_rrs["Date"].astype(str).tolist()

    # === Match target bands ===
    rrs_wavelengths = []
    rrs_cols = []
    for col in df_rrs.columns:
        try:
            wl = float(col)
            rrs_wavelengths.append(wl)
            rrs_cols.append(col)
        except Exception:
            continue

    band_cols = []
    matched_bands = []
    for target_band in selected_bands:
        diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
        min_diff = min(diffs)
        if min_diff > max_allowed_diff:
            raise ValueError(
                f"Target wavelength {target_band} nm cannot be matched, error {min_diff:.2f} nm exceeds the allowed range"
            )
        best_idx = diffs.index(min_diff)
        band_cols.append(rrs_cols[best_idx])
        matched_bands.append(rrs_wavelengths[best_idx])

    print(
        f"\n✅ Band matching successful, {len(selected_bands)} target bands in total, {len(band_cols)} columns actually extracted"
    )
    print(f"Original number of test samples: {df_rrs.shape[0]}\n")

    # === Extract Rrs and target parameter (without differencing for now) ===
    Rrs_array = df_rrs[band_cols].values.astype(float)
    target_array = df_param[[target_parameter]].values.astype(float).flatten()

    # === Key: Remove rows with NaN/Inf before differencing ===
    mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_target_ok = np.isfinite(target_array)
    mask_ok = mask_inputs_ok & mask_target_ok
    if not np.any(mask_ok):
        raise ValueError("❌ No valid samples (NaN/Inf found in input or target).")
    dropped = int(len(mask_ok) - mask_ok.sum())
    if dropped > 0:
        print(
            f"⚠️ Dropped {dropped} invalid samples (containing NaN/Inf) before differencing"
        )

    Rrs_array = Rrs_array[mask_ok]
    target_array = target_array[mask_ok]
    sample_ids = [sid for sid, keep in zip(sample_ids, mask_ok) if keep]
    sample_dates = [d for d, keep in zip(sample_dates, mask_ok) if keep]

    # === Preprocessing before differencing (optional) ===
    if diff_before_norm:
        Rrs_array = np.diff(Rrs_array, axis=1)

    # === Apply MinMaxScaler to [1, 10] for each sample ===
    scalers_Rrs_test = [MinMaxScaler((1, 10)) for _ in range(Rrs_array.shape[0])]
    Rrs_normalized = np.array(
        [
            scalers_Rrs_test[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    # === Post-processing after differencing (optional) ===
    if diff_after_norm:
        Rrs_normalized = np.diff(Rrs_normalized, axis=1)

    # === Transform target value to log10(x + log_offset) ===
    target_transformed = np.log10(target_array + log_offset)

    # === Construct DataLoader ===
    Rrs_tensor = torch.tensor(Rrs_normalized, dtype=torch.float32)
    target_tensor = torch.tensor(target_transformed.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(Rrs_tensor, target_tensor)
    test_dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = target_tensor.shape[1]

    return test_dl, input_dim, output_dim, sample_ids, sample_dates


def load_real_data_with_BR_LH(
    excel_path,
    csv_path,
    selected_bands,
    split_ratio=0.7,
    seed=42,
    target_parameter="PC",
    lower_quantile=0.0,
    upper_quantile=1.0,
    pc_upper_limit=1000,
    log_offset=0.01,
):
    rounded_bands = [int(round(b)) for b in selected_bands]
    band_cols = [f"Rrs_{b}" for b in rounded_bands]
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")
    df_rrs_selected = df_rrs[["GLORIA_ID"] + band_cols]
    df_param_selected = df_param[["GLORIA_ID", target_parameter]]
    df_merged = pd.merge(
        df_rrs_selected, df_param_selected, on="GLORIA_ID", how="inner"
    )

    # Step 1: Remove NaNs
    mask_rrs_valid = df_merged[band_cols].notna().all(axis=1)
    mask_target_valid = df_merged[target_parameter].notna()
    df_filtered = df_merged[mask_rrs_valid & mask_target_valid].reset_index(drop=True)
    print(f"✅ Samples after removing NaNs: {len(df_filtered)}")

    # Step 2: Quantile clipping
    lower = df_filtered[target_parameter].quantile(lower_quantile)
    upper = df_filtered[target_parameter].quantile(upper_quantile)
    df_filtered = df_filtered[
        (df_filtered[target_parameter] >= lower)
        & (df_filtered[target_parameter] <= upper)
    ]

    # Step 3: Clip PC upper limit
    if target_parameter.upper() == "PC":
        df_filtered = df_filtered[df_filtered[target_parameter] <= pc_upper_limit]
    df_filtered = df_filtered.reset_index(drop=True)
    print(f"✅ Samples after quantile and PC filtering: {len(df_filtered)}")

    # Step 4: Normalize Rrs (sample-wise)
    all_sample_ids = df_filtered["GLORIA_ID"].astype(str).tolist()
    Rrs_array = df_filtered[band_cols].values
    scalers_Rrs_real = [MinMaxScaler((1, 10)) for _ in range(Rrs_array.shape[0])]
    Rrs_normalized = np.array(
        [
            scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    # Step 5: Transform target parameter
    param_array = df_filtered[[target_parameter]].values
    param_transformed = np.log10(param_array + log_offset)

    # Step 6: Load BR/LH definitions
    df_combos = pd.read_csv(csv_path)
    br_rows = df_combos[df_combos["Type"] == "BR"]
    lh_rows = df_combos[df_combos["Type"] == "LH"]

    # Step 7: Compute BR features
    br_features = []
    for _, row in br_rows.iterrows():
        b1 = int(row["Detail1"])
        b2 = int(row["Detail2"])
        col1 = f"Rrs_{b1}"
        col2 = f"Rrs_{b2}"
        if col1 in df_filtered.columns and col2 in df_filtered.columns:
            ratio = (df_filtered[col1] / df_filtered[col2]).values.reshape(-1, 1)
            br_features.append(ratio)
    br_array = (
        np.hstack(br_features) if br_features else np.zeros((len(df_filtered), 0))
    )

    # Step 8: Compute LH features (三波段线性基线法)
    lh_features = []
    for _, row in lh_rows.iterrows():
        wl_c = int(row["Detail1"])  # center band
        offset = int(row["Detail2"])  # ±d
        wl1 = wl_c - offset
        wl2 = wl_c + offset

        col_c = f"Rrs_{wl_c}"
        col_1 = f"Rrs_{wl1}"
        col_2 = f"Rrs_{wl2}"

        if (
            col_c in df_filtered.columns
            and col_1 in df_filtered.columns
            and col_2 in df_filtered.columns
        ):
            rrs_c = df_filtered[col_c].values
            rrs_1 = df_filtered[col_1].values
            rrs_2 = df_filtered[col_2].values

            baseline = rrs_1 + ((wl_c - wl1) / (wl2 - wl1)) * (rrs_2 - rrs_1)
            lh_value = (rrs_c - baseline).reshape(-1, 1)
            lh_features.append(lh_value)
    lh_array = (
        np.hstack(lh_features) if lh_features else np.zeros((len(df_filtered), 0))
    )

    # Step 9: Combine all features
    X_combined = np.hstack([Rrs_normalized, br_array, lh_array])
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)
    y_tensor = torch.tensor(param_transformed, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Step 10: Train/test split
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_size = int(split_ratio * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_ids = [all_sample_ids[i] for i in train_indices]
    test_ids = [all_sample_ids[i] for i in test_indices]

    train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    input_dim = X_tensor.shape[1]
    output_dim = y_tensor.shape[1]

    return (train_dl, test_dl, input_dim, output_dim, train_ids, test_ids)


def load_real_test_with_BR_LH(
    excel_path,
    csv_path,
    selected_bands,
    target_parameter="PC",
    pc_upper_limit=1000,
    log_offset=0.01,
):
    # === Load Excel ===
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    # ✅ 强制把 Rrs 列名中是数字的部分转为 int（如 '440' -> 440）
    df_rrs.columns = [
        int(c) if str(c).replace(".", "", 1).isdigit() else c for c in df_rrs.columns
    ]

    # === Extract ID and Date ===
    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = df_rrs["Date"].astype(str).tolist()

    # === Match Bands ===
    rrs_wavelengths = [c for c in df_rrs.columns if isinstance(c, int)]
    band_cols = []
    for b in selected_bands:
        diffs = [abs(w - b) for w in rrs_wavelengths]
        best_idx = np.argmin(diffs)
        if diffs[best_idx] > 1.0:
            raise ValueError(
                f"❌ No match for band {b} nm (closest diff = {diffs[best_idx]:.2f})"
            )
        band_cols.append(rrs_wavelengths[best_idx])

    df_rrs_selected = df_rrs[band_cols].copy()
    df_param_selected = df_param[[target_parameter]].copy()

    # === Remove NaN/Inf ===
    Rrs_array = df_rrs_selected.values.astype(float)
    target_array = df_param_selected.values.astype(float).flatten()
    mask_rrs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_param_ok = np.isfinite(target_array)
    mask_ok = mask_rrs_ok & mask_param_ok
    Rrs_array = Rrs_array[mask_ok]
    target_array = target_array[mask_ok]
    sample_ids = [sid for sid, keep in zip(sample_ids, mask_ok) if keep]
    sample_dates = [d for d, keep in zip(sample_dates, mask_ok) if keep]
    print(f"✅ Valid samples after NaN filtering: {len(sample_ids)}")

    # === Clip PC upper limit ===
    if target_parameter.upper() == "PC":
        mask_pc = target_array <= pc_upper_limit
        Rrs_array = Rrs_array[mask_pc]
        target_array = target_array[mask_pc]
        sample_ids = [sid for sid, keep in zip(sample_ids, mask_pc) if keep]
        sample_dates = [d for d, keep in zip(sample_dates, mask_pc) if keep]
        print(f"✅ Samples after PC clipping (≤ {pc_upper_limit}): {len(sample_ids)}")

    # === Normalize Rrs to [1, 10] per sample ===
    scalers_Rrs = [MinMaxScaler((1, 10)) for _ in range(Rrs_array.shape[0])]
    Rrs_normalized = np.array(
        [
            scalers_Rrs[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    # === Load BR / LH definition ===
    df_combos = pd.read_csv(csv_path)
    br_rows = df_combos[df_combos["Type"] == "BR"]
    lh_rows = df_combos[df_combos["Type"] == "LH"]
    df_rrs_full = df_rrs.loc[mask_ok].reset_index(drop=True)

    if target_parameter.upper() == "PC":
        df_rrs_full = df_rrs_full[mask_pc].reset_index(drop=True)

    # === Compute BR features ===
    br_features = []
    br_skipped = []
    for _, row in br_rows.iterrows():
        b1 = int(row["Detail1"])
        b2 = int(row["Detail2"])
        if b1 in df_rrs_full.columns and b2 in df_rrs_full.columns:
            ratio = (df_rrs_full[b1] / df_rrs_full[b2]).values.reshape(-1, 1)
            br_features.append(ratio)
        else:
            br_skipped.append((b1, b2))
    br_array = (
        np.hstack(br_features) if br_features else np.zeros((len(df_rrs_full), 0))
    )
    print(f"✅ BR 特征数量：{br_array.shape[1]}")
    print(f"❌ 跳过的 BR 组合（共 {len(br_skipped)} 个）：")
    for b1, b2 in br_skipped[:20]:
        print(f"  - {b1} / {b2}")
    if len(br_skipped) > 20:
        print("  ...")

    # === Compute LH features ===
    lh_features = []
    lh_skipped = []
    for _, row in lh_rows.iterrows():
        wl_c = int(row["Detail1"])
        offset = int(row["Detail2"])
        wl1 = wl_c - offset
        wl2 = wl_c + offset
        if (
            wl_c in df_rrs_full.columns
            and wl1 in df_rrs_full.columns
            and wl2 in df_rrs_full.columns
        ):
            rrs_c = df_rrs_full[wl_c].values
            rrs_1 = df_rrs_full[wl1].values
            rrs_2 = df_rrs_full[wl2].values
            baseline = rrs_1 + ((wl_c - wl1) / (wl2 - wl1)) * (rrs_2 - rrs_1)
            lh_value = (rrs_c - baseline).reshape(-1, 1)
            lh_features.append(lh_value)
        else:
            lh_skipped.append((wl_c, offset))
    lh_array = (
        np.hstack(lh_features) if lh_features else np.zeros((len(df_rrs_full), 0))
    )
    print(f"✅ LH 特征数量：{lh_array.shape[1]}")
    print(f"❌ 跳过的 LH 组合（共 {len(lh_skipped)} 个）：")
    for wl_c, offset in lh_skipped[:20]:
        print(f"  - Center: {wl_c}, Offset: ±{offset}")
    if len(lh_skipped) > 20:
        print("  ...")

    # === Combine all features ===
    X_combined = np.hstack([Rrs_normalized, br_array, lh_array])
    y_transformed = np.log10(target_array + log_offset).reshape(-1, 1)

    # === Construct DataLoader ===
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)
    y_tensor = torch.tensor(y_transformed, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    test_dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    input_dim = X_combined.shape[1]
    output_dim = 1

    print(f"✅ 总特征数量（Rrs + BR + LH）：{input_dim}")
    print(f"✅ Test DataLoader length: {len(test_dl)}")

    return test_dl, input_dim, output_dim, sample_ids, sample_dates


def build_real_train_test_by_date(
    excel_path,
    selected_bands,
    target_parameter="TSS",
    seed=42,
    max_allowed_diff=1.0,
    diff_before_norm=False,
    diff_after_norm=False,
    log_offset=0.01,
    train_batch_size=1024,
):
    """
    Returns
    -------
    train_loader
    test_loader
    test_metadata dict:
        sample_ids, dates, lat, lon, area
    """

    # ================= READ =================
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if len(df_rrs) != len(df_param):
        raise ValueError("Rrs and parameter row mismatch")

    df_rrs["Date"] = pd.to_datetime(df_rrs["Date"])
    df_param["Date"] = pd.to_datetime(df_param["Date"])

    # ================= SPLIT BY DATE =================
    # ================= SPLIT HALF WITHIN EACH DATE =================
    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    for date, group in df_rrs.groupby("Date"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        half = len(indices) // 2

        train_indices.extend(indices[:half])
        test_indices.extend(indices[half:])

    # build train/test
    df_rrs_train = df_rrs.loc[train_indices].reset_index(drop=True)
    df_param_train = df_param.loc[train_indices].reset_index(drop=True)

    df_rrs_test = df_rrs.loc[test_indices].reset_index(drop=True)
    df_param_test = df_param.loc[test_indices].reset_index(drop=True)

    # ================= INNER BUILDER =================
    def build_loader(
        df_rrs_local, df_param_local, shuffle, batch_size, return_meta=False
    ):

        # ---- metadata (before drop NaN) ----
        sample_ids = df_rrs_local["Site Label"].astype(str).tolist()
        sample_dates = df_rrs_local["Date"].astype(str).tolist()
        lats = df_rrs_local["Lat"].astype(float).tolist()
        lons = df_rrs_local["Long"].astype(float).tolist()
        areas = df_param_local["Area"].astype(str).tolist()

        # ---- find wavelength columns ----
        rrs_wavelengths = []
        rrs_cols = []
        for col in df_rrs_local.columns:
            try:
                wl = float(col)
                rrs_wavelengths.append(wl)
                rrs_cols.append(col)
            except:
                continue

        # ---- band match ----
        band_cols = []
        for target_band in selected_bands:
            diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
            min_diff = min(diffs)
            if min_diff > max_allowed_diff:
                raise ValueError(f"Band {target_band} unmatched ({min_diff:.2f} nm)")
            band_cols.append(rrs_cols[diffs.index(min_diff)])

        # ---- extract arrays ----
        Rrs_array = df_rrs_local[band_cols].values.astype(float)
        target_array = df_param_local[[target_parameter]].values.astype(float).flatten()

        # ---- remove NaN BEFORE normalization ----
        mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
        mask_target_ok = np.isfinite(target_array)
        mask_ok = mask_inputs_ok & mask_target_ok

        Rrs_array = Rrs_array[mask_ok]
        target_array = target_array[mask_ok]

        sample_ids_f = [x for x, m in zip(sample_ids, mask_ok) if m]
        sample_dates_f = [x for x, m in zip(sample_dates, mask_ok) if m]
        lats_f = [x for x, m in zip(lats, mask_ok) if m]
        lons_f = [x for x, m in zip(lons, mask_ok) if m]
        areas_f = [x for x, m in zip(areas, mask_ok) if m]

        # ---- diff before norm ----
        if diff_before_norm:
            Rrs_array = np.diff(Rrs_array, axis=1)

        # ---- per-sample minmax ----
        scalers = [MinMaxScaler((1, 10)) for _ in range(len(Rrs_array))]
        Rrs_norm = np.array(
            [
                scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
                for i, row in enumerate(Rrs_array)
            ]
        )

        # ---- diff after norm ----
        if diff_after_norm:
            Rrs_norm = np.diff(Rrs_norm, axis=1)

        # ---- target transform ----
        target_transformed = np.log10(target_array + log_offset)

        # ---- tensors ----
        X = torch.tensor(Rrs_norm, dtype=torch.float32)
        y = torch.tensor(target_transformed.reshape(-1, 1), dtype=torch.float32)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if return_meta:
            meta = {
                "sample_ids": sample_ids_f,
                "dates": sample_dates_f,
                "lat": lats_f,
                "lon": lons_f,
                "area": areas_f,
            }
            return loader, meta
        else:
            return loader

    # ================= BUILD =================
    train_loader = build_loader(
        df_rrs_train, df_param_train, shuffle=True, batch_size=train_batch_size
    )
    test_loader, test_meta = build_loader(
        df_rrs_test,
        df_param_test,
        shuffle=False,
        batch_size=len(df_rrs_test),
        return_meta=True,
    )

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Test samples : {len(test_loader.dataset)}")

    return train_loader, test_loader, test_meta


def build_real_test_loader(
    excel_path,
    selected_bands,
    target_parameter="TSS",
    max_allowed_diff=1.0,
    diff_before_norm=False,
    diff_after_norm=False,
    log_offset=0.01,
    batch_size=4096,
):

    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler

    # ================= READ =================
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if len(df_rrs) != len(df_param):
        raise ValueError("Rrs and parameter row mismatch")

    df_rrs["Date"] = pd.to_datetime(df_rrs["Date"])

    # ================= META =================
    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = df_rrs["Date"].astype(str).tolist()
    lats = df_rrs["Lat"].astype(float).tolist()
    lons = df_rrs["Long"].astype(float).tolist()
    areas = df_param["Area"].astype(str).tolist()

    # ================= FIND WAVELENGTH COLUMNS =================
    rrs_wavelengths = []
    rrs_cols = []

    for col in df_rrs.columns:
        try:
            wl = float(col)
            rrs_wavelengths.append(wl)
            rrs_cols.append(col)
        except:
            continue

    # ================= BAND MATCH =================
    band_cols = []

    for target_band in selected_bands:

        diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
        min_diff = min(diffs)

        if min_diff > max_allowed_diff:
            raise ValueError(f"Band {target_band} unmatched ({min_diff:.2f} nm)")

        band_cols.append(rrs_cols[diffs.index(min_diff)])

    # ================= EXTRACT ARRAYS =================
    Rrs_array = df_rrs[band_cols].values.astype(float)
    target_array = df_param[[target_parameter]].values.astype(float).flatten()

    # ================= REMOVE NaN =================
    mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_target_ok = np.isfinite(target_array)

    mask_ok = mask_inputs_ok & mask_target_ok

    Rrs_array = Rrs_array[mask_ok]
    target_array = target_array[mask_ok]

    sample_ids_f = [x for x, m in zip(sample_ids, mask_ok) if m]
    sample_dates_f = [x for x, m in zip(sample_dates, mask_ok) if m]
    lats_f = [x for x, m in zip(lats, mask_ok) if m]
    lons_f = [x for x, m in zip(lons, mask_ok) if m]
    areas_f = [x for x, m in zip(areas, mask_ok) if m]

    # ================= DIFF BEFORE NORM =================
    if diff_before_norm:
        Rrs_array = np.diff(Rrs_array, axis=1)

    # ================= PER-SAMPLE MINMAX =================
    scalers = [MinMaxScaler((1, 10)) for _ in range(len(Rrs_array))]

    Rrs_norm = np.array(
        [
            scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    # ================= DIFF AFTER NORM =================
    if diff_after_norm:
        Rrs_norm = np.diff(Rrs_norm, axis=1)

    # ================= TARGET TRANSFORM =================
    target_transformed = np.log10(target_array + log_offset)

    # ================= TENSOR =================
    X = torch.tensor(Rrs_norm, dtype=torch.float32)
    y = torch.tensor(target_transformed.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X, y)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ================= META =================
    test_meta = {
        "sample_ids": sample_ids_f,
        "dates": sample_dates_f,
        "lat": lats_f,
        "lon": lons_f,
        "area": areas_f,
    }

    print(f"\nTotal samples: {len(dataset)}")

    return test_loader, test_meta


def build_real_test_loader_robust(
    excel_path,
    selected_bands,
    scaler_Rrs,
    scalers_dict,
    target_parameter="SPM",
    max_allowed_diff=1.0,
    use_diff=False,
    batch_size=4096,
):

    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # ================= READ =================
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if len(df_rrs) != len(df_param):
        raise ValueError("Rrs and parameter row mismatch")

    # ================= META =================
    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = pd.to_datetime(df_rrs["Date"]).astype(str).tolist()
    lats = df_rrs["Lat"].astype(float).tolist()
    lons = df_rrs["Long"].astype(float).tolist()
    areas = df_param["Area"].astype(str).tolist()

    # ================= WAVELENGTH MATCH =================
    rrs_wavelengths = []
    rrs_cols = []

    for col in df_rrs.columns:
        try:
            wl = float(col)
            rrs_wavelengths.append(wl)
            rrs_cols.append(col)
        except:
            continue

    band_cols = []

    for target_band in selected_bands:

        diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
        min_diff = min(diffs)

        if min_diff > max_allowed_diff:
            raise ValueError(f"Band {target_band} unmatched ({min_diff:.2f} nm)")

        band_cols.append(rrs_cols[diffs.index(min_diff)])

    # ================= ARRAYS =================
    Rrs_array = df_rrs[band_cols].values.astype(float)
    target_array = df_param[[target_parameter]].values.astype(float).flatten()

    # ================= REMOVE NaN =================
    mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_target_ok = np.isfinite(target_array)
    mask_ok = mask_inputs_ok & mask_target_ok

    Rrs_array = Rrs_array[mask_ok]
    target_array = target_array[mask_ok]

    sample_ids_f = [x for x, m in zip(sample_ids, mask_ok) if m]
    sample_dates_f = [x for x, m in zip(sample_dates, mask_ok) if m]
    lats_f = [x for x, m in zip(lats, mask_ok) if m]
    lons_f = [x for x, m in zip(lons, mask_ok) if m]
    areas_f = [x for x, m in zip(areas, mask_ok) if m]

    # ================= DIFF =================
    if use_diff:
        Rrs_array = np.diff(Rrs_array, axis=1)

    # ================= Rrs ROBUST NORMALIZATION =================
    Rrs_tensor = torch.tensor(Rrs_array, dtype=torch.float32)
    Rrs_norm = scaler_Rrs.transform(Rrs_tensor).numpy()

    # ================= TARGET TRANSFORM =================
    log_scaler = scalers_dict["log"]
    robust_scaler = scalers_dict["robust"]

    param_log = log_scaler.transform(
        torch.tensor(target_array.reshape(-1, 1), dtype=torch.float32)
    )

    target_transformed = robust_scaler.transform(param_log).numpy()

    # ================= TENSOR =================
    X = torch.tensor(Rrs_norm, dtype=torch.float32)
    y = torch.tensor(target_transformed.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X, y)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ================= META =================
    test_meta = {
        "sample_ids": sample_ids_f,
        "dates": sample_dates_f,
        "lat": lats_f,
        "lon": lons_f,
        "area": areas_f,
    }

    print(f"\nTotal samples: {len(dataset)}")

    return test_loader, test_meta
