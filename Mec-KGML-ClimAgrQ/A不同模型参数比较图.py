# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 1. File path
# =========================
input_path = r"D:\实验\毕业论文\第四章\不同学习的R2-RMSE-MAE变化\参数汇总表.xlsx"
output_dir = os.path.dirname(input_path)
output_path = os.path.join(output_dir, "model_metric_barplot_group.png")

# =========================
# 2. Plot settings
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.size"] = 14

# =========================
# 3. Read data
# =========================
df = pd.read_excel(input_path)

# 如果第一列没有列名，通常会被读成 Unnamed: 0
first_col = df.columns[0]
df = df.rename(columns={first_col: "Metric"})
df["Metric"] = df["Metric"].astype(str).str.strip()

# 统一 R-squared 的显示名称
metric_name_map = {
    "R-squared": "R²",
    "R2": "R²",
    "R²": "R²",
    "RMSE": "RMSE",
    "MAE": "MAE",
}
df["Metric"] = df["Metric"].replace(metric_name_map)

# 转成长表
plot_df = df.melt(id_vars="Metric", var_name="Model", value_name="Value")
plot_df["Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
plot_df = plot_df.dropna(subset=["Value"])

model_order = [c for c in df.columns if c != "Metric"]
metric_order = ["R²", "RMSE", "MAE"]

# =========================
# 4. Helper functions
# =========================
def make_gradient_colors(n, color_start, color_end):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_start, color_end])
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def get_padded_ylim(values, metric):
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    vrange = vmax - vmin

    if vrange <= 1e-12:
        pad = max(abs(vmax) * 0.05, 0.01)
    else:
        pad = vrange * 0.25

    ymin = vmin - pad
    ymax = vmax + pad

    # 不同指标使用不同起点，突出差异
    if metric == "R²":
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
    elif metric == "RMSE":
        ymin = max(0.0, ymin)
    elif metric == "MAE":
        ymin = max(0.0, ymin)

    return ymin, ymax


# 三个指标使用不同但协调的渐变色
color_schemes = {
    "R²": ("#d8ecf3", "#2b6f9e"),      # 蓝色系
    "RMSE": ("#f8e4c2", "#c9781f"),    # 暖橙色系
    "MAE": ("#dcefd8", "#4d8b4a"),     # 绿色系
}

# =========================
# 5. Plot grouped figure
# =========================
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 5.2),
    constrained_layout=False
)

bar_width = 0.68
x = np.arange(len(model_order))

for ax, metric in zip(axes, metric_order):
    sub = plot_df[plot_df["Metric"] == metric].copy()
    sub = sub.set_index("Model").reindex(model_order).reset_index()

    values = sub["Value"].values
    colors = make_gradient_colors(
        len(model_order),
        color_schemes[metric][0],
        color_schemes[metric][1]
    )

    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=1.1
    )

    ymin, ymax = get_padded_ylim(values, metric)
    ax.set_ylim(ymin, ymax)

    #ax.set_title(metric, fontsize=20, fontweight="bold", pad=12)
    ax.set_ylabel(metric, fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=13)

    ax.grid(axis="y", linestyle="--", alpha=0.30, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=13)

    # 数值标注
    value_range = ymax - ymin
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + value_range * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=90
        )

# 整体布局
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.24, top=0.86, wspace=0.28)
fig.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved to: {output_path}")