import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter


def calculate_metrics(predictions, actuals, threshold=0.8):
    """
    Calculate epsilon, beta and additional metrics (NRMSE, RMSLE, MAPE, Bias, MAE).

    :param predictions: array-like, predicted values
    :param actuals: array-like, actual values
    :param threshold: float, relative error threshold
    :return: epsilon, beta, nrmse, rmsle, mape, bias, mae
    """
    eps = 1e-10  # small constant to avoid division by zero

    predictions = np.where(predictions <= eps, eps, predictions)
    actuals = np.where(actuals <= eps, eps, actuals)
    filtered_predictions = predictions
    filtered_actuals = actuals

    # Calculate epsilon and beta
    log_ratios = np.log10(filtered_predictions / filtered_actuals)
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (10**Y - 1)
    beta = 100 * np.sign(Z) * (10 ** np.abs(Z) - 1)

    # NRMSE: RMSE normalized by range (max - min)
    rmse = np.sqrt(np.mean((filtered_predictions - filtered_actuals) ** 2))
    nrmse = rmse / (np.max(filtered_actuals) - np.min(filtered_actuals) + eps)

    rmsle = np.sqrt(
        np.mean(
            (np.log10(filtered_predictions + 1) - np.log10(filtered_actuals + 1)) ** 2
        )
    )
    mape = 100 * np.median(
        np.abs((filtered_predictions - filtered_actuals) / filtered_actuals)
    )
    bias = 10 ** (np.mean(np.log10(filtered_predictions) - np.log10(filtered_actuals)))
    mae = 10 ** np.mean(
        np.abs(np.log10(filtered_predictions) - np.log10(filtered_actuals))
    )

    return epsilon, beta, nrmse, rmsle, mape, bias, mae


def plot_results(
    predictions_rescaled,
    actuals_rescaled,
    save_dir,
    threshold=10,
    mode="test",
    xlim=(-4, 4),
    ylim=(-4, 4),
):
    os.makedirs(save_dir, exist_ok=True)

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    log_actuals = np.log10(np.where(actuals == 0, 1e-10, actuals))
    log_predictions = np.log10(np.where(predictions == 0, 1e-10, predictions))

    mask = np.abs(log_predictions - log_actuals) < threshold
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]

    filtered_log_actual = np.log10(
        np.where(filtered_actuals == 0, 1e-10, filtered_actuals)
    )
    filtered_log_prediction = np.log10(
        np.where(filtered_predictions == 0, 1e-10, filtered_predictions)
    )

    epsilon, beta, nrmse, rmsle, mape, bias, mae = calculate_metrics(
        filtered_predictions, filtered_actuals, threshold
    )

    valid_mask = np.isfinite(filtered_log_actual) & np.isfinite(filtered_log_prediction)
    slope, intercept = np.polyfit(
        filtered_log_actual[valid_mask], filtered_log_prediction[valid_mask], 1
    )
    x = np.array([xlim[0], xlim[1]])
    y = slope * x + intercept

    plt.figure(figsize=(6, 6))

    # Regression line
    plt.plot(x, y, linestyle="--", color="blue", linewidth=0.8)
    # 1:1 line
    plt.plot(xlim, ylim, linestyle="-", color="black", linewidth=0.8)

    # Scatter & KDE
    sns.scatterplot(x=log_actuals, y=log_predictions, alpha=0.5)
    sns.kdeplot(
        x=filtered_log_actual,
        y=filtered_log_prediction,
        levels=3,
        color="black",
        fill=False,
        linewidths=0.8,
    )

    plt.xlabel("Actual Values", fontsize=16, fontname="Ubuntu")
    plt.ylabel("Predicted Values", fontsize=16, fontname="Ubuntu")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, which="both", ls="--")

    plt.legend(
        title=(
            f"MAE = {mae:.2f}, NRMSE = {nrmse:.2f}, RMSLE = {rmsle:.2f}\n"
            f"Bias = {bias:.2f}, Slope = {slope:.2f}\n"
            f"MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%"
        ),
        fontsize=12,
        title_fontsize=10,
        prop={"family": "Ubuntu"},
    )

    plt.xticks(fontsize=14, fontname="Ubuntu")
    plt.yticks(fontsize=14, fontname="Ubuntu")

    png_path = os.path.join(save_dir, f"{mode}_plot.png")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"✅ Saved and displayed: {png_path}")


def plot_results_with_density(
    predictions_rescaled,
    actuals_rescaled,
    save_dir,
    threshold=10,
    mode="test_density",
    xlim=(-4, 4),
    ylim=(-4, 4),
    cmap="viridis",
    tick_min=-4,
    tick_max=4,
    tick_step=1,
):
    os.makedirs(save_dir, exist_ok=True)

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    # Log10 transform (avoid non-positive values)
    eps = 1e-10
    log_actuals = np.log10(np.where(actuals <= 0, eps, actuals))
    log_predictions = np.log10(np.where(predictions <= 0, eps, predictions))

    # Optional threshold filtering
    mask = np.abs(log_predictions - log_actuals) < threshold
    log_a_f, log_p_f = log_actuals[mask], log_predictions[mask]
    a_f, p_f = actuals[mask], predictions[mask]

    # Density estimation with Gaussian KDE
    xy = np.vstack([log_a_f, log_p_f])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    log_a_f, log_p_f, z = log_a_f[idx], log_p_f[idx], z[idx]
    a_f, p_f = a_f[idx], p_f[idx]

    # Calculate metrics
    epsilon, beta, nrmse, rmsle, mape, bias, mae = calculate_metrics(
        p_f, a_f, threshold
    )

    # Linear regression (in log-log space)
    valid_mask = np.isfinite(log_a_f) & np.isfinite(log_p_f)
    slope, intercept = np.polyfit(log_a_f[valid_mask], log_p_f[valid_mask], 1)

    # === Plot ===
    plt.figure(figsize=(8, 6), dpi=300)

    # 1:1 reference line (do not add to legend)
    plt.plot(
        [xlim[0], xlim[1]],
        [xlim[0], xlim[1]],
        linestyle="-",
        color="black",
        linewidth=0.9,
    )

    # Regression line (do not add to legend)
    xs = np.array([xlim[0], xlim[1]])
    ys = slope * xs + intercept
    plt.plot(xs, ys, linestyle="--", color="blue", linewidth=0.9)

    # Scatter points colored by density
    sc = plt.scatter(log_a_f, log_p_f, c=z, s=30, cmap=cmap, alpha=1, edgecolors="none")

    # Colorbar for density
    cbar = plt.colorbar(sc, fraction=0.06, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    # Axis ticks (shown as powers of 10)
    ax = plt.gca()
    ticks = np.arange(tick_min, tick_max + 1e-9, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    formatter = FuncFormatter(lambda val, pos: f"$10^{{{val:.1f}}}$")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=16)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, ls="--", alpha=0.5)

    # Metrics in legend (only title shown)
    plt.legend(
        title=(
            f"MAE = {mae:.2f}, NRMSE = {nrmse:.2f}\n"
            f"RMSLE = {rmsle:.2f}, Bias = {bias:.2f}\n"
            f"MAPE = {mape:.2f}%, Slope = {slope:.2f}\n"
            f"ε = {epsilon:.2f}%, β = {beta:.2f}%"
        ),
        fontsize=12,
        title_fontsize=10,
        frameon=True,
    )

    # Save figure as PNG
    png_path = os.path.join(save_dir, f"{mode}.png")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"✅ Saved and displayed PNG: {png_path}")


def save_results_to_excelV2(
    ids, actuals, predictions, file_path, dates=None, lat=None, lon=None, area=None
):
    """
    Save prediction results to Excel.

    Columns order:
    ID | Date | Lat | Lon | Area | Actual | Predicted
    Missing fields will be automatically skipped.
    """

    data = {"ID": ids}

    if dates is not None:
        data["Date"] = dates

    if lat is not None:
        data["Lat"] = lat

    if lon is not None:
        data["Lon"] = lon

    if area is not None:
        data["Area"] = area

    data["Actual"] = actuals
    data["Predicted"] = predictions

    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)


def save_results_from_excel_for_testV2(
    predictions,
    actuals,
    sample_ids,
    dates,
    original_excel_path,
    save_dir,
    lat=None,
    lon=None,
    area=None,
):

    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(original_excel_path)
    dataset_name = os.path.splitext(filename)[0]

    save_path = os.path.join(save_dir, f"{dataset_name}.xlsx")

    save_results_to_excelV2(
        ids=sample_ids,
        actuals=actuals,
        predictions=predictions,
        file_path=save_path,
        dates=dates,
        lat=lat,
        lon=lon,
        area=area,
    )


def save_results_to_excel(ids, actuals, predictions, file_path, dates=None):
    """
    Save prediction results to an Excel file.
    - If dates are included, output ID, Date, Actual, Predicted;
    - Otherwise, output ID, Actual, Predicted.
    """
    if dates is not None:
        df = pd.DataFrame(
            {"ID": ids, "Date": dates, "Actual": actuals, "Predicted": predictions}
        )
    else:
        df = pd.DataFrame({"ID": ids, "Actual": actuals, "Predicted": predictions})

    df.to_excel(file_path, index=False)


def save_results_from_excel_for_test(
    predictions, actuals, sample_ids, dates, original_excel_path, save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(original_excel_path)
    dataset_name = os.path.splitext(filename)[0]

    save_results_to_excel(
        sample_ids,
        actuals,
        predictions,
        os.path.join(save_dir, f"{dataset_name}.xlsx"),
        dates=dates,
    )
