# -*- coding: utf-8 -*-
"""
M0 与 M3 独立预测能力综合比较代码

当前版本：
1. 原有综合指标仍不纳入 R2；新增单独输出预测结果 R2 图。
2. 不计算和显示 MAPE、sMAPE。
3. 不计算和显示 NRMSE。
4. 连续值指标、四等级商品等级指标、三分类风险等级指标，输出列名统一使用空格分隔，不使用下划线。
5. 保留连续预测图：
   01 Observed vs predicted
   07 Residual density
   09 Absolute error boxplot
   21 Absolute error by observed value group
   30 Integrated M0 M3 comparison panel
   38 Observed、M0、M3 箱线图，配对样本连线，均值和标准差标注，均值 95% CI 连线
6. 保留四等级商品等级预测图：
   31 商品等级混淆矩阵 M0
   32 商品等级混淆矩阵 M3
   33 商品等级总体分类指标柱状图，不含 Within one grade accuracy
   34 商品等级数量分布图
   35 商品等级逐等级 F1 图
   36 商品等级误差分布图
   37 商品等级逐等级 Accuracy 图
7. 新增三分类垩白风险等级：
   x ≤ 5        Low chalkiness risk
   5 < x ≤ 8    Moderate chalkiness risk
   x > 8        High chalkiness risk
8. 新增三分类风险等级图：
   41 垩白风险等级混淆矩阵 M0
   42 垩白风险等级混淆矩阵 M3
   43 垩白风险等级总体判别指标图
   44 垩白风险等级逐等级 F1 图
   45 垩白风险等级逐等级 Accuracy 图
   46 垩白风险等级数量分布图
9. 新增单独 R2 图：
   00 Prediction R2 M0 M3
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)

from scipy import stats

warnings.filterwarnings("ignore")


# =========================================================
# 1. 路径设置
# =========================================================
INPUT_FILE = r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\实测-M0-M3.xlsx"

OUTPUT_DIR = os.path.join(
    os.path.dirname(INPUT_FILE),
    "M0_M3预测能力综合比较_美化版_空格指标名_含风险等级_箱线配对图"
)
TABLE_DIR = os.path.join(OUTPUT_DIR, "01_结果表格")
FIG_DIR = os.path.join(OUTPUT_DIR, "02_结果图件")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =========================================================
# 2. 图形风格设置
# =========================================================
plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 15
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 17
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["legend.title_fontsize"] = 13

sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    rc={
        "axes.edgecolor": "#2B2B2B",
        "axes.linewidth": 1.15,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "grid.color": "#B0B0B0"
    }
)

MODEL_PALETTE = {
    "M0": "#4E79A7",
    "M3": "#F28E2B"
}

MODEL_LIGHT_PALETTE = {
    "M0": "#A9C4E2",
    "M3": "#F8C38B"
}

MODEL_MARKERS = {
    "M0": "o",
    "M3": "s"
}

VALUE_PALETTE = {
    "Observed": "#333333",
    "M0": "#4E79A7",
    "M3": "#F28E2B"
}

VALUE_LIGHT_PALETTE = {
    "Observed": "#BDBDBD",
    "M0": "#A9C4E2",
    "M3": "#F8C38B"
}

SERIES_PALETTE = {
    "Observed count": "#333333",
    "M0 predicted count": "#4E79A7",
    "M3 predicted count": "#F28E2B"
}

BAR_WIDTH = 0.52
EDGE_COLOR = "#2B2B2B"


# =========================================================
# 3. 基础函数
# =========================================================
def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_first_nonempty_sheet(excel_file):
    xls = pd.ExcelFile(excel_file)
    for sheet in xls.sheet_names:
        temp = pd.read_excel(excel_file, sheet_name=sheet)
        if temp.shape[0] > 0 and temp.shape[1] >= 3:
            return sheet
    raise ValueError("未找到包含至少3列有效数据的工作表。")


def lin_ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true, ddof=1)
    var_pred = np.var(y_pred, ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]

    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return np.nan

    return 2 * cov / denom


def willmott_d(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum(
        (np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2
    )

    if denominator <= 1e-12:
        return np.nan

    return 1 - numerator / denominator


def nse_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator <= 1e-12:
        return np.nan

    return 1 - numerator / denominator


def rpd_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sd = np.std(y_true, ddof=1)

    if rmse <= 1e-12:
        return np.nan

    return sd / rmse


def calculate_metrics(y_true, y_pred, model_name):
    """
    回归预测能力指标。
    不计算 R2，不计算 MAPE、sMAPE 和 NRMSE。
    输出列名统一使用空格。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_true - y_pred
    prediction_error = y_pred - y_true
    abs_error = np.abs(residual)

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)

    slope, intercept, reg_r, reg_p, reg_se = stats.linregress(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    out = {
        "Model": model_name,
        "n": len(y_true),

        "Observed mean": np.mean(y_true),
        "Observed SD": np.std(y_true, ddof=1),
        "Observed min": np.min(y_true),
        "Observed max": np.max(y_true),

        "Predicted mean": np.mean(y_pred),
        "Predicted SD": np.std(y_pred, ddof=1),
        "Predicted min": np.min(y_pred),
        "Predicted max": np.max(y_pred),

        "Mean bias pred minus obs": np.mean(prediction_error),
        "Median bias pred minus obs": np.median(prediction_error),

        "RMSE": rmse,
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mae,
        "Median AE": np.median(abs_error),
        "Max AE": np.max(abs_error),

        "Pearson r": pearson_r,
        "Pearson p": pearson_p,
        "Spearman r": spearman_r,
        "Spearman p": spearman_p,
        "Kendall tau": kendall_tau,
        "Kendall p": kendall_p,

        "Lin CCC": lin_ccc(y_true, y_pred),
        "Willmott d": willmott_d(y_true, y_pred),
        "NSE": nse_score(y_true, y_pred),
        "RPD": rpd_score(y_true, y_pred),

        "Regression slope pred on obs": slope,
        "Regression intercept pred on obs": intercept,
        "Regression p": reg_p,

        "Residual mean obs minus pred": np.mean(residual),
        "Residual median obs minus pred": np.median(residual),
        "Residual SD": np.std(residual, ddof=1)
    }

    return out


def improvement_rate(m0_value, m3_value, metric):
    higher_better = [
        "Pearson r",
        "Spearman r",
        "Kendall tau",
        "Lin CCC",
        "Willmott d",
        "NSE",
        "RPD",
        "Accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
        "Weighted precision",
        "Weighted recall",
        "Weighted F1",
        "Cohen kappa",
        "Within one grade accuracy"
    ]

    lower_better = [
        "RMSE",
        "MSE",
        "MAE",
        "Median AE",
        "Max AE",
        "Residual SD",
        "Mean absolute grade error",
        "Mean absolute risk error"
    ]

    if pd.isna(m0_value) or pd.isna(m3_value):
        return np.nan

    if abs(m0_value) <= 1e-12:
        return np.nan

    if metric in higher_better:
        return (m3_value - m0_value) / abs(m0_value) * 100

    if metric in lower_better:
        return (m0_value - m3_value) / abs(m0_value) * 100

    return np.nan


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def format_ax(ax, grid_axis="y", keep_top=False, keep_right=False):
    if not keep_top:
        ax.spines["top"].set_visible(False)
    if not keep_right:
        ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)

    ax.tick_params(axis="both", labelsize=14, width=1.0, color=EDGE_COLOR)

    if grid_axis == "y":
        ax.grid(axis="y", linestyle="--", alpha=0.28, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    elif grid_axis == "x":
        ax.grid(axis="x", linestyle="--", alpha=0.28, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    elif grid_axis == "both":
        ax.grid(axis="both", linestyle="--", alpha=0.25, linewidth=0.75)
    else:
        ax.grid(False)

    return ax


def shrink_bar_width(ax, width=BAR_WIDTH):
    for patch in ax.patches:
        try:
            old_width = patch.get_width()
            if old_width <= 0:
                continue
            center = patch.get_x() + old_width / 2
            new_width = min(width, old_width)
            patch.set_width(new_width)
            patch.set_x(center - new_width / 2)
            patch.set_linewidth(1.1)
            patch.set_edgecolor(EDGE_COLOR)
        except Exception:
            pass
    return ax


def add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3):
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=padding)
        except Exception:
            pass
    return ax


def set_focus_ylim(ax, values, lower_margin=0.30, upper_margin=0.45, min_zero=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return ax

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        pad = abs(vmax) * 0.05 if abs(vmax) > 1e-12 else 1.0
        lower = vmin - pad
        upper = vmax + pad
    else:
        span = vmax - vmin
        lower = vmin - span * lower_margin
        upper = vmax + span * upper_margin

    if min_zero:
        lower = max(0, lower)

    ax.set_ylim(lower, upper)
    return ax


def set_count_ylim(ax, values, top_margin=0.18):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return ax
    ymax = float(np.max(values))
    ax.set_ylim(0, ymax * (1 + top_margin) if ymax > 0 else 1)
    return ax


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean_value = np.mean(values)
    sd_value = np.std(values, ddof=1) if n > 1 else 0.0

    if n > 1:
        se = sd_value / np.sqrt(n)
        t_value = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean_value - t_value * se
        ci_high = mean_value + t_value * se
    else:
        ci_low = mean_value
        ci_high = mean_value

    return mean_value, sd_value, ci_low, ci_high


# =========================================================
# 4. 四等级商品等级函数
# =========================================================
GRADE_LABELS = ["Grade 1", "Grade 2", "Grade 3", "Out of grade"]

GRADE_ORDER = {
    "Grade 1": 1,
    "Grade 2": 2,
    "Grade 3": 3,
    "Out of grade": 4
}


def quality_grade(x):
    if pd.isna(x):
        return np.nan

    x = float(x)

    if x <= 2:
        return "Grade 1"
    elif x <= 5:
        return "Grade 2"
    elif x <= 8:
        return "Grade 3"
    else:
        return "Out of grade"


def calculate_grade_metrics(actual_grade, pred_grade, model_name):
    actual_grade = pd.Series(actual_grade).astype(str)
    pred_grade = pd.Series(pred_grade).astype(str)

    acc = accuracy_score(actual_grade, pred_grade)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        actual_grade,
        pred_grade,
        labels=GRADE_LABELS,
        average="macro",
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        actual_grade,
        pred_grade,
        labels=GRADE_LABELS,
        average="weighted",
        zero_division=0
    )

    kappa = cohen_kappa_score(actual_grade, pred_grade, labels=GRADE_LABELS)

    actual_code = actual_grade.map(GRADE_ORDER).astype(float)
    pred_code = pred_grade.map(GRADE_ORDER).astype(float)

    grade_error = pred_code - actual_code
    abs_grade_error = np.abs(grade_error)

    within_one_grade_acc = np.mean(abs_grade_error <= 1)

    out = {
        "Model": model_name,
        "n": len(actual_grade),
        "Accuracy": acc,
        "Macro precision": precision_macro,
        "Macro recall": recall_macro,
        "Macro F1": f1_macro,
        "Weighted precision": precision_weighted,
        "Weighted recall": recall_weighted,
        "Weighted F1": f1_weighted,
        "Cohen kappa": kappa,
        "Within one grade accuracy": within_one_grade_acc,
        "Mean grade error pred minus obs": np.mean(grade_error),
        "Mean absolute grade error": np.mean(abs_grade_error),
        "Exact match n": int(np.sum(actual_grade.values == pred_grade.values)),
        "Within one grade n": int(np.sum(abs_grade_error <= 1))
    }

    return out


def grade_classification_report(actual_grade, pred_grade, model_name):
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_grade,
        pred_grade,
        labels=GRADE_LABELS,
        zero_division=0
    )

    report_df = pd.DataFrame({
        "Model": model_name,
        "Grade": GRADE_LABELS,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support
    })

    return report_df


def grade_confusion_matrix_array(actual_grade, pred_grade):
    return confusion_matrix(actual_grade, pred_grade, labels=GRADE_LABELS)


def grade_confusion_matrix_df(actual_grade, pred_grade, model_name):
    cm = grade_confusion_matrix_array(actual_grade, pred_grade)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Observed {g}" for g in GRADE_LABELS],
        columns=[f"Predicted {g}" for g in GRADE_LABELS]
    )

    cm_df.insert(0, "Model", model_name)

    return cm_df


# =========================================================
# 5. 三分类垩白风险等级函数
# =========================================================
RISK_LABELS = [
    "Low chalkiness risk",
    "Moderate chalkiness risk",
    "High chalkiness risk"
]

RISK_ORDER = {
    "Low chalkiness risk": 1,
    "Moderate chalkiness risk": 2,
    "High chalkiness risk": 3
}


def chalkiness_risk_grade(x):
    if pd.isna(x):
        return np.nan

    x = float(x)

    if x <= 5:
        return "Low chalkiness risk"
    elif x <= 8:
        return "Moderate chalkiness risk"
    else:
        return "High chalkiness risk"


def calculate_risk_metrics(actual_risk, pred_risk, model_name):
    actual_risk = pd.Series(actual_risk).astype(str)
    pred_risk = pd.Series(pred_risk).astype(str)

    acc = accuracy_score(actual_risk, pred_risk)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        actual_risk,
        pred_risk,
        labels=RISK_LABELS,
        average="macro",
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        actual_risk,
        pred_risk,
        labels=RISK_LABELS,
        average="weighted",
        zero_division=0
    )

    kappa = cohen_kappa_score(actual_risk, pred_risk, labels=RISK_LABELS)

    actual_code = actual_risk.map(RISK_ORDER).astype(float)
    pred_code = pred_risk.map(RISK_ORDER).astype(float)

    risk_error = pred_code - actual_code
    abs_risk_error = np.abs(risk_error)

    out = {
        "Model": model_name,
        "n": len(actual_risk),
        "Accuracy": acc,
        "Macro precision": precision_macro,
        "Macro recall": recall_macro,
        "Macro F1": f1_macro,
        "Weighted precision": precision_weighted,
        "Weighted recall": recall_weighted,
        "Weighted F1": f1_weighted,
        "Cohen kappa": kappa,
        "Mean risk error pred minus obs": np.mean(risk_error),
        "Mean absolute risk error": np.mean(abs_risk_error),
        "Exact match n": int(np.sum(actual_risk.values == pred_risk.values))
    }

    return out


def risk_classification_report(actual_risk, pred_risk, model_name):
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_risk,
        pred_risk,
        labels=RISK_LABELS,
        zero_division=0
    )

    report_df = pd.DataFrame({
        "Model": model_name,
        "Risk class": RISK_LABELS,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support
    })

    return report_df


def risk_confusion_matrix_array(actual_risk, pred_risk):
    return confusion_matrix(actual_risk, pred_risk, labels=RISK_LABELS)


def risk_confusion_matrix_df(actual_risk, pred_risk, model_name):
    cm = risk_confusion_matrix_array(actual_risk, pred_risk)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Observed {g}" for g in RISK_LABELS],
        columns=[f"Predicted {g}" for g in RISK_LABELS]
    )

    cm_df.insert(0, "Model", model_name)

    return cm_df


# =========================================================
# 6. 读取数据
# =========================================================
sheet_name = find_first_nonempty_sheet(INPUT_FILE)
df_raw = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
df_raw = clean_columns(df_raw)

if df_raw.shape[1] < 3:
    raise ValueError("输入表至少需要3列：实测值、M0预测值、M3预测值。")

actual_col = df_raw.columns[0]
m0_col = df_raw.columns[1]
m3_col = df_raw.columns[2]

df = df_raw.iloc[:, :3].copy()
df.columns = ["Actual", "M0", "M3"]

for col in ["Actual", "M0", "M3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Actual", "M0", "M3"]).reset_index(drop=True)
df["Sample ID"] = np.arange(1, len(df) + 1)

if len(df) < 3:
    raise ValueError("有效样本数少于3，无法进行稳定的模型比较。")

print("读取完成：")
print(f"工作表：{sheet_name}")
print(f"原始列名：{actual_col}, {m0_col}, {m3_col}")
print(f"有效样本数：{len(df)}")


# =========================================================
# 7. 构建长表与残差表
# =========================================================
long_df = pd.concat([
    pd.DataFrame({
        "Sample ID": df["Sample ID"],
        "Actual": df["Actual"],
        "Predicted": df["M0"],
        "Model": "M0"
    }),
    pd.DataFrame({
        "Sample ID": df["Sample ID"],
        "Actual": df["Actual"],
        "Predicted": df["M3"],
        "Model": "M3"
    })
], axis=0, ignore_index=True)

long_df["Residual"] = long_df["Actual"] - long_df["Predicted"]
long_df["Prediction error"] = long_df["Predicted"] - long_df["Actual"]
long_df["Absolute error"] = np.abs(long_df["Residual"])
long_df["Squared error"] = long_df["Residual"] ** 2

wide_diag = df.copy()
wide_diag["M0 residual"] = wide_diag["Actual"] - wide_diag["M0"]
wide_diag["M3 residual"] = wide_diag["Actual"] - wide_diag["M3"]
wide_diag["M0 prediction error"] = wide_diag["M0"] - wide_diag["Actual"]
wide_diag["M3 prediction error"] = wide_diag["M3"] - wide_diag["Actual"]
wide_diag["M0 absolute error"] = np.abs(wide_diag["M0 residual"])
wide_diag["M3 absolute error"] = np.abs(wide_diag["M3 residual"])
wide_diag["M0 squared error"] = wide_diag["M0 residual"] ** 2
wide_diag["M3 squared error"] = wide_diag["M3 residual"] ** 2
wide_diag["Absolute error reduction M0 minus M3"] = (
    wide_diag["M0 absolute error"] - wide_diag["M3 absolute error"]
)
wide_diag["Squared error reduction M0 minus M3"] = (
    wide_diag["M0 squared error"] - wide_diag["M3 squared error"]
)
wide_diag["M3 better absolute error"] = wide_diag["M3 absolute error"] < wide_diag["M0 absolute error"]
wide_diag["M3 better squared error"] = wide_diag["M3 squared error"] < wide_diag["M0 squared error"]


# =========================================================
# 8. 总体回归指标
# =========================================================
metrics_df = pd.DataFrame([
    calculate_metrics(df["Actual"], df["M0"], "M0"),
    calculate_metrics(df["Actual"], df["M3"], "M3")
])

improvement_records = []
for metric in metrics_df.columns:
    if metric in ["Model", "n"]:
        continue

    m0_value = metrics_df.loc[metrics_df["Model"] == "M0", metric].values[0]
    m3_value = metrics_df.loc[metrics_df["Model"] == "M3", metric].values[0]

    improvement_records.append({
        "Metric": metric,
        "M0": m0_value,
        "M3": m3_value,
        "M3 minus M0": m3_value - m0_value if pd.notna(m0_value) and pd.notna(m3_value) else np.nan,
        "Improvement percent": improvement_rate(m0_value, m3_value, metric)
    })

improvement_df = pd.DataFrame(improvement_records)

# =========================================================
# 8.1 单独计算 R2，仅用于单独 R2 图和表格
# =========================================================
r2_df = pd.DataFrame([
    {
        "Model": "M0",
        "R2": r2_score(df["Actual"], df["M0"])
    },
    {
        "Model": "M3",
        "R2": r2_score(df["Actual"], df["M3"])
    }
])


# =========================================================
# 9. 分实测值区间比较
# =========================================================
try:
    wide_diag["Actual group"] = pd.qcut(
        wide_diag["Actual"],
        q=4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        duplicates="drop"
    )
except Exception:
    wide_diag["Actual group"] = pd.cut(
        wide_diag["Actual"],
        bins=4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"]
    )

group_records = []
for group_name, sub in wide_diag.groupby("Actual group"):
    if len(sub) < 2:
        continue

    for model in ["M0", "M3"]:
        pred = sub[model]
        y = sub["Actual"]
        rec = calculate_metrics(y, pred, model)
        rec["Actual group"] = str(group_name)
        group_records.append(rec)

group_metrics_df = pd.DataFrame(group_records)


# =========================================================
# 10. 配对统计检验
# =========================================================
m0_abs = wide_diag["M0 absolute error"].values
m3_abs = wide_diag["M3 absolute error"].values
m0_sq = wide_diag["M0 squared error"].values
m3_sq = wide_diag["M3 squared error"].values
m0_res = wide_diag["M0 residual"].values
m3_res = wide_diag["M3 residual"].values

test_records = []


def add_test(name, stat, p, note):
    test_records.append({
        "Comparison": name,
        "Statistic": stat,
        "p value": p,
        "Note": note
    })


try:
    stat, p = stats.ttest_rel(m0_abs, m3_abs)
    add_test(
        "Paired t test of absolute error",
        stat,
        p,
        "p < 0.05 indicates significant difference in absolute error between M0 and M3."
    )
except Exception as e:
    add_test("Paired t test of absolute error", np.nan, np.nan, str(e))

try:
    stat, p = stats.wilcoxon(m0_abs, m3_abs)
    add_test(
        "Wilcoxon signed rank test of absolute error",
        stat,
        p,
        "Non parametric paired comparison of absolute error."
    )
except Exception as e:
    add_test("Wilcoxon signed rank test of absolute error", np.nan, np.nan, str(e))

try:
    stat, p = stats.ttest_rel(m0_sq, m3_sq)
    add_test(
        "Paired t test of squared error",
        stat,
        p,
        "p < 0.05 indicates significant difference in squared error between M0 and M3."
    )
except Exception as e:
    add_test("Paired t test of squared error", np.nan, np.nan, str(e))

try:
    stat, p = stats.wilcoxon(m0_sq, m3_sq)
    add_test(
        "Wilcoxon signed rank test of squared error",
        stat,
        p,
        "Non parametric paired comparison of squared error."
    )
except Exception as e:
    add_test("Wilcoxon signed rank test of squared error", np.nan, np.nan, str(e))

try:
    stat, p = stats.levene(m0_res, m3_res)
    add_test(
        "Levene test of residual variance",
        stat,
        p,
        "p < 0.05 indicates different residual variance."
    )
except Exception as e:
    add_test("Levene test of residual variance", np.nan, np.nan, str(e))

test_df = pd.DataFrame(test_records)


# =========================================================
# 11. Bootstrap 置信区间
# =========================================================
def bootstrap_metric_ci(y_true, pred, metric_func, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    values = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        try:
            values.append(metric_func(y_true[idx], pred[idx]))
        except Exception:
            values.append(np.nan)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    return np.nanmean(values), np.nanpercentile(values, 2.5), np.nanpercentile(values, 97.5)


bootstrap_metrics = {
    "RMSE": lambda y, p: np.sqrt(mean_squared_error(y, p)),
    "MAE": lambda y, p: mean_absolute_error(y, p),
    "Lin CCC": lambda y, p: lin_ccc(y, p),
    "Willmott d": lambda y, p: willmott_d(y, p),
    "RPD": lambda y, p: rpd_score(y, p)
}

boot_records = []
for model in ["M0", "M3"]:
    pred = df[model].values
    for metric_name, func in bootstrap_metrics.items():
        mean_v, low_v, high_v = bootstrap_metric_ci(df["Actual"].values, pred, func)
        boot_records.append({
            "Model": model,
            "Metric": metric_name,
            "Bootstrap mean": mean_v,
            "CI 2.5 percent": low_v,
            "CI 97.5 percent": high_v
        })

bootstrap_df = pd.DataFrame(boot_records)


# =========================================================
# 12. 四等级商品等级预测能力评价
# =========================================================
wide_diag["Observed grade"] = wide_diag["Actual"].apply(quality_grade)
wide_diag["M0 predicted grade"] = wide_diag["M0"].apply(quality_grade)
wide_diag["M3 predicted grade"] = wide_diag["M3"].apply(quality_grade)

wide_diag["Observed grade code"] = wide_diag["Observed grade"].map(GRADE_ORDER)
wide_diag["M0 predicted grade code"] = wide_diag["M0 predicted grade"].map(GRADE_ORDER)
wide_diag["M3 predicted grade code"] = wide_diag["M3 predicted grade"].map(GRADE_ORDER)

wide_diag["M0 grade error"] = wide_diag["M0 predicted grade code"] - wide_diag["Observed grade code"]
wide_diag["M3 grade error"] = wide_diag["M3 predicted grade code"] - wide_diag["Observed grade code"]

wide_diag["M0 absolute grade error"] = np.abs(wide_diag["M0 grade error"])
wide_diag["M3 absolute grade error"] = np.abs(wide_diag["M3 grade error"])

wide_diag["M3 better grade prediction"] = wide_diag["M3 absolute grade error"] < wide_diag["M0 absolute grade error"]
wide_diag["M3 same grade prediction"] = wide_diag["M3 absolute grade error"] == wide_diag["M0 absolute grade error"]
wide_diag["M3 worse grade prediction"] = wide_diag["M3 absolute grade error"] > wide_diag["M0 absolute grade error"]

grade_metric_df = pd.DataFrame([
    calculate_grade_metrics(wide_diag["Observed grade"], wide_diag["M0 predicted grade"], "M0"),
    calculate_grade_metrics(wide_diag["Observed grade"], wide_diag["M3 predicted grade"], "M3")
])

grade_improvement_records = []
for metric in grade_metric_df.columns:
    if metric in ["Model", "n"]:
        continue

    m0_value = grade_metric_df.loc[grade_metric_df["Model"] == "M0", metric].values[0]
    m3_value = grade_metric_df.loc[grade_metric_df["Model"] == "M3", metric].values[0]

    grade_improvement_records.append({
        "Metric": metric,
        "M0": m0_value,
        "M3": m3_value,
        "M3 minus M0": m3_value - m0_value if pd.notna(m0_value) and pd.notna(m3_value) else np.nan,
        "Improvement percent": improvement_rate(m0_value, m3_value, metric)
    })

grade_improvement_df = pd.DataFrame(grade_improvement_records)

grade_report_df = pd.concat([
    grade_classification_report(wide_diag["Observed grade"], wide_diag["M0 predicted grade"], "M0"),
    grade_classification_report(wide_diag["Observed grade"], wide_diag["M3 predicted grade"], "M3")
], axis=0, ignore_index=True)

grade_accuracy_records = []

for model, pred_col in [
    ("M0", "M0 predicted grade"),
    ("M3", "M3 predicted grade")
]:
    for grade in GRADE_LABELS:
        sub = wide_diag[wide_diag["Observed grade"] == grade].copy()
        support = len(sub)

        if support == 0:
            acc_value = np.nan
            correct_n = 0
        else:
            correct_n = int((sub[pred_col] == grade).sum())
            acc_value = correct_n / support

        grade_accuracy_records.append({
            "Model": model,
            "Grade": grade,
            "Grade specific accuracy": acc_value,
            "Correct n": correct_n,
            "Support": support
        })

grade_accuracy_df = pd.DataFrame(grade_accuracy_records)

grade_cm_m0 = grade_confusion_matrix_df(wide_diag["Observed grade"], wide_diag["M0 predicted grade"], "M0")
grade_cm_m3 = grade_confusion_matrix_df(wide_diag["Observed grade"], wide_diag["M3 predicted grade"], "M3")
grade_cm_df = pd.concat([grade_cm_m0, grade_cm_m3], axis=0, ignore_index=True)

grade_count_df = pd.DataFrame({
    "Grade": GRADE_LABELS,
    "Observed count": [np.sum(wide_diag["Observed grade"] == g) for g in GRADE_LABELS],
    "M0 predicted count": [np.sum(wide_diag["M0 predicted grade"] == g) for g in GRADE_LABELS],
    "M3 predicted count": [np.sum(wide_diag["M3 predicted grade"] == g) for g in GRADE_LABELS]
})

grade_pair_summary = pd.DataFrame([{
    "n": len(wide_diag),
    "M3 better than M0 n": int(wide_diag["M3 better grade prediction"].sum()),
    "M3 same as M0 n": int(wide_diag["M3 same grade prediction"].sum()),
    "M3 worse than M0 n": int(wide_diag["M3 worse grade prediction"].sum()),
    "M3 better than M0 percent": wide_diag["M3 better grade prediction"].mean() * 100,
    "M3 same as M0 percent": wide_diag["M3 same grade prediction"].mean() * 100,
    "M3 worse than M0 percent": wide_diag["M3 worse grade prediction"].mean() * 100
}])


# =========================================================
# 13. 三分类垩白风险等级评价
# =========================================================
wide_diag["Observed risk"] = wide_diag["Actual"].apply(chalkiness_risk_grade)
wide_diag["M0 predicted risk"] = wide_diag["M0"].apply(chalkiness_risk_grade)
wide_diag["M3 predicted risk"] = wide_diag["M3"].apply(chalkiness_risk_grade)

wide_diag["Observed risk code"] = wide_diag["Observed risk"].map(RISK_ORDER)
wide_diag["M0 predicted risk code"] = wide_diag["M0 predicted risk"].map(RISK_ORDER)
wide_diag["M3 predicted risk code"] = wide_diag["M3 predicted risk"].map(RISK_ORDER)

wide_diag["M0 risk error"] = wide_diag["M0 predicted risk code"] - wide_diag["Observed risk code"]
wide_diag["M3 risk error"] = wide_diag["M3 predicted risk code"] - wide_diag["Observed risk code"]

wide_diag["M0 absolute risk error"] = np.abs(wide_diag["M0 risk error"])
wide_diag["M3 absolute risk error"] = np.abs(wide_diag["M3 risk error"])

risk_metric_df = pd.DataFrame([
    calculate_risk_metrics(wide_diag["Observed risk"], wide_diag["M0 predicted risk"], "M0"),
    calculate_risk_metrics(wide_diag["Observed risk"], wide_diag["M3 predicted risk"], "M3")
])

risk_improvement_records = []
for metric in risk_metric_df.columns:
    if metric in ["Model", "n"]:
        continue

    m0_value = risk_metric_df.loc[risk_metric_df["Model"] == "M0", metric].values[0]
    m3_value = risk_metric_df.loc[risk_metric_df["Model"] == "M3", metric].values[0]

    risk_improvement_records.append({
        "Metric": metric,
        "M0": m0_value,
        "M3": m3_value,
        "M3 minus M0": m3_value - m0_value if pd.notna(m0_value) and pd.notna(m3_value) else np.nan,
        "Improvement percent": improvement_rate(m0_value, m3_value, metric)
    })

risk_improvement_df = pd.DataFrame(risk_improvement_records)

risk_report_df = pd.concat([
    risk_classification_report(wide_diag["Observed risk"], wide_diag["M0 predicted risk"], "M0"),
    risk_classification_report(wide_diag["Observed risk"], wide_diag["M3 predicted risk"], "M3")
], axis=0, ignore_index=True)

risk_accuracy_records = []

for model, pred_col in [
    ("M0", "M0 predicted risk"),
    ("M3", "M3 predicted risk")
]:
    for risk_class in RISK_LABELS:
        sub = wide_diag[wide_diag["Observed risk"] == risk_class].copy()
        support = len(sub)

        if support == 0:
            acc_value = np.nan
            correct_n = 0
        else:
            correct_n = int((sub[pred_col] == risk_class).sum())
            acc_value = correct_n / support

        risk_accuracy_records.append({
            "Model": model,
            "Risk class": risk_class,
            "Risk specific accuracy": acc_value,
            "Correct n": correct_n,
            "Support": support
        })

risk_accuracy_df = pd.DataFrame(risk_accuracy_records)

risk_cm_m0 = risk_confusion_matrix_df(wide_diag["Observed risk"], wide_diag["M0 predicted risk"], "M0")
risk_cm_m3 = risk_confusion_matrix_df(wide_diag["Observed risk"], wide_diag["M3 predicted risk"], "M3")
risk_cm_df = pd.concat([risk_cm_m0, risk_cm_m3], axis=0, ignore_index=True)

risk_count_df = pd.DataFrame({
    "Risk class": RISK_LABELS,
    "Observed count": [np.sum(wide_diag["Observed risk"] == g) for g in RISK_LABELS],
    "M0 predicted count": [np.sum(wide_diag["M0 predicted risk"] == g) for g in RISK_LABELS],
    "M3 predicted count": [np.sum(wide_diag["M3 predicted risk"] == g) for g in RISK_LABELS]
})


# =========================================================
# 14. 输出表格
# =========================================================
metrics_df.to_excel(os.path.join(TABLE_DIR, "01 总体预测能力指标 无R2 无MAPE 无NRMSE M0 vs M3.xlsx"), index=False)
improvement_df.to_excel(os.path.join(TABLE_DIR, "02 M3相对M0改进率 无R2 无MAPE 无NRMSE.xlsx"), index=False)
r2_df.to_excel(os.path.join(TABLE_DIR, "00 单独R2指标 M0 vs M3.xlsx"), index=False)
wide_diag.to_excel(os.path.join(TABLE_DIR, "03 逐样本残差等级风险诊断表.xlsx"), index=False)
long_df.to_excel(os.path.join(TABLE_DIR, "04 长格式预测残差表.xlsx"), index=False)
group_metrics_df.to_excel(os.path.join(TABLE_DIR, "05 按实测值分位数分组指标 无R2 无MAPE 无NRMSE.xlsx"), index=False)
test_df.to_excel(os.path.join(TABLE_DIR, "06 配对统计检验结果.xlsx"), index=False)
bootstrap_df.to_excel(os.path.join(TABLE_DIR, "07 Bootstrap置信区间 无R2 无MAPE 无NRMSE.xlsx"), index=False)

grade_metric_df.to_excel(os.path.join(TABLE_DIR, "08 四等级商品等级预测总体指标.xlsx"), index=False)
grade_improvement_df.to_excel(os.path.join(TABLE_DIR, "09 四等级商品等级M3相对M0改进率.xlsx"), index=False)
grade_report_df.to_excel(os.path.join(TABLE_DIR, "10 四等级商品等级逐等级分类指标.xlsx"), index=False)
grade_cm_df.to_excel(os.path.join(TABLE_DIR, "11 四等级商品等级混淆矩阵.xlsx"), index=False)
grade_count_df.to_excel(os.path.join(TABLE_DIR, "12 四等级商品等级数量分布.xlsx"), index=False)
grade_pair_summary.to_excel(os.path.join(TABLE_DIR, "13 四等级商品等级逐样本改进情况汇总.xlsx"), index=False)
grade_accuracy_df.to_excel(os.path.join(TABLE_DIR, "14 四等级商品等级逐等级Accuracy.xlsx"), index=False)

risk_metric_df.to_excel(os.path.join(TABLE_DIR, "15 三分类垩白风险等级预测总体指标.xlsx"), index=False)
risk_improvement_df.to_excel(os.path.join(TABLE_DIR, "16 三分类垩白风险等级M3相对M0改进率.xlsx"), index=False)
risk_report_df.to_excel(os.path.join(TABLE_DIR, "17 三分类垩白风险等级逐等级分类指标.xlsx"), index=False)
risk_cm_df.to_excel(os.path.join(TABLE_DIR, "18 三分类垩白风险等级混淆矩阵.xlsx"), index=False)
risk_accuracy_df.to_excel(os.path.join(TABLE_DIR, "19 三分类垩白风险等级逐等级Accuracy.xlsx"), index=False)
risk_count_df.to_excel(os.path.join(TABLE_DIR, "20 三分类垩白风险等级数量分布.xlsx"), index=False)


# =========================================================
# 15. 连续预测能力绘图
# =========================================================

# ---------------------------------------------------------
# 00 单独 R2 柱状图
# ---------------------------------------------------------
plt.figure(figsize=(5.8, 4.8))
ax = sns.barplot(
    data=r2_df,
    x="Model",
    y="R2",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.48,
    legend=False
)

shrink_bar_width(ax, width=0.32)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.axhline(0, color="#1F1F1F", linestyle="--", linewidth=1.05)
ax.set_xlabel("")
ax.set_ylabel("R²")

r2_values = r2_df["R2"].replace([np.inf, -np.inf], np.nan).dropna().values
if len(r2_values) > 0:
    r2_min = float(np.nanmin(r2_values))
    r2_max = float(np.nanmax(r2_values))
    if np.isclose(r2_min, r2_max):
        pad = 0.10 if abs(r2_max) < 1 else abs(r2_max) * 0.10
    else:
        pad = (r2_max - r2_min) * 0.35
    ymin = min(0, r2_min - pad)
    ymax = r2_max + pad
    ax.set_ylim(ymin, ymax)

format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "00 Prediction R2 M0 M3.png"))


# ---------------------------------------------------------
# 01 观测值与预测值散点图
# ---------------------------------------------------------
plt.figure(figsize=(6.8, 6.2))
ax = sns.scatterplot(
    data=long_df,
    x="Actual",
    y="Predicted",
    hue="Model",
    style="Model",
    palette=MODEL_PALETTE,
    markers=MODEL_MARKERS,
    s=96,
    edgecolor=EDGE_COLOR,
    linewidth=0.65,
    alpha=0.86
)

min_v = min(long_df["Actual"].min(), long_df["Predicted"].min())
max_v = max(long_df["Actual"].max(), long_df["Predicted"].max())
pad = (max_v - min_v) * 0.06

plt.plot(
    [min_v - pad, max_v + pad],
    [min_v - pad, max_v + pad],
    color="#1F1F1F",
    linestyle="--",
    linewidth=1.45,
    label="1:1 line"
)

for model in ["M0", "M3"]:
    sub = long_df[long_df["Model"] == model]
    slope, intercept, r_value, p_value, std_err = stats.linregress(sub["Actual"], sub["Predicted"])
    x_line = np.linspace(min_v, max_v, 100)
    y_line = slope * x_line + intercept
    plt.plot(
        x_line,
        y_line,
        linewidth=2.35,
        color=MODEL_PALETTE[model],
        alpha=0.95
    )

ax.set_xlabel("")
ax.set_ylabel("Predicted value")
# 图顶部标题已按要求删除
ax.legend(frameon=True, loc="best", edgecolor=EDGE_COLOR)
format_ax(ax, grid_axis="both")
save_fig(os.path.join(FIG_DIR, "01 Observed vs predicted M0 M3.png"))


# ---------------------------------------------------------
# 07 残差核密度图
# ---------------------------------------------------------
plt.figure(figsize=(7.0, 5.0))
ax = sns.kdeplot(
    data=long_df,
    x="Residual",
    hue="Model",
    palette=MODEL_PALETTE,
    fill=True,
    alpha=0.22,
    linewidth=2.45,
    common_norm=False
)

ax.axvline(0, color="#1F1F1F", linestyle="--", linewidth=1.45)
ax.set_xlabel("")
ax.set_ylabel("Density")
# 图顶部标题已按要求删除
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "07 Residual density.png"))


# ---------------------------------------------------------
# 09 绝对误差箱线图
# ---------------------------------------------------------
plt.figure(figsize=(5.8, 4.8))
ax = sns.boxplot(
    data=long_df,
    x="Model",
    y="Absolute error",
    hue="Model",
    palette=MODEL_LIGHT_PALETTE,
    width=0.42,
    fliersize=2.8,
    linewidth=1.25,
    legend=False,
    boxprops={"edgecolor": EDGE_COLOR},
    medianprops={"color": EDGE_COLOR, "linewidth": 1.5},
    whiskerprops={"color": EDGE_COLOR, "linewidth": 1.15},
    capprops={"color": EDGE_COLOR, "linewidth": 1.15}
)

sns.stripplot(
    data=long_df,
    x="Model",
    y="Absolute error",
    hue="Model",
    palette=MODEL_PALETTE,
    size=3.5,
    alpha=0.58,
    jitter=0.16,
    edgecolor=EDGE_COLOR,
    linewidth=0.35,
    legend=False,
    ax=ax
)

ax.set_xlabel("")
ax.set_ylabel("Absolute error")
# 图顶部标题已按要求删除
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "09 Absolute error boxplot.png"))


# ---------------------------------------------------------
# 21 按实测值分位数组的误差比较
# ---------------------------------------------------------
group_error_long = pd.concat([
    pd.DataFrame({
        "Actual group": wide_diag["Actual group"].astype(str),
        "Absolute error": wide_diag["M0 absolute error"],
        "Model": "M0"
    }),
    pd.DataFrame({
        "Actual group": wide_diag["Actual group"].astype(str),
        "Absolute error": wide_diag["M3 absolute error"],
        "Model": "M3"
    })
], axis=0, ignore_index=True)

plt.figure(figsize=(8.4, 5.2))
ax = sns.boxplot(
    data=group_error_long,
    x="Actual group",
    y="Absolute error",
    hue="Model",
    palette=MODEL_LIGHT_PALETTE,
    width=0.56,
    linewidth=1.15,
    fliersize=2.5,
    boxprops={"edgecolor": EDGE_COLOR},
    medianprops={"color": EDGE_COLOR, "linewidth": 1.4},
    whiskerprops={"color": EDGE_COLOR, "linewidth": 1.1},
    capprops={"color": EDGE_COLOR, "linewidth": 1.1}
)

ax.set_xlabel("")
ax.set_ylabel("Absolute error")
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=12)
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "21 Absolute error by observed value group.png"))


# ---------------------------------------------------------
# 30 综合图，不设置 title，主指标只放 RMSE 和 MAE
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2))

ax = axes[0, 0]
sns.scatterplot(
    data=long_df,
    x="Actual",
    y="Predicted",
    hue="Model",
    style="Model",
    palette=MODEL_PALETTE,
    markers=MODEL_MARKERS,
    s=72,
    edgecolor=EDGE_COLOR,
    linewidth=0.55,
    alpha=0.86,
    ax=ax
)

min_v = min(long_df["Actual"].min(), long_df["Predicted"].min())
max_v = max(long_df["Actual"].max(), long_df["Predicted"].max())
pad = (max_v - min_v) * 0.06

ax.plot(
    [min_v - pad, max_v + pad],
    [min_v - pad, max_v + pad],
    color="#1F1F1F",
    linestyle="--",
    linewidth=1.25
)

ax.set_xlabel("")
ax.set_ylabel("Predicted value")
ax.legend(frameon=True, edgecolor=EDGE_COLOR, loc="best")
format_ax(ax, grid_axis="both")

ax = axes[0, 1]
main_df = metrics_df[["Model", "RMSE", "MAE"]].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Value"
)

sns.barplot(
    data=main_df,
    x="Metric",
    y="Value",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.56,
    ax=ax
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("Value")
set_focus_ylim(ax, main_df["Value"].values, lower_margin=0.55, upper_margin=0.65, min_zero=False)
ax.legend(frameon=True, edgecolor=EDGE_COLOR, loc="best")
format_ax(ax, grid_axis="y")

ax = axes[1, 0]
sns.kdeplot(
    data=long_df,
    x="Residual",
    hue="Model",
    palette=MODEL_PALETTE,
    fill=True,
    alpha=0.20,
    linewidth=2.25,
    common_norm=False,
    ax=ax
)

ax.axvline(0, color="#1F1F1F", linestyle="--", linewidth=1.25)
ax.set_xlabel("")
ax.set_ylabel("Density")
format_ax(ax, grid_axis="y")

ax = axes[1, 1]
sns.boxplot(
    data=long_df,
    x="Model",
    y="Absolute error",
    hue="Model",
    palette=MODEL_LIGHT_PALETTE,
    width=0.42,
    linewidth=1.15,
    fliersize=2.5,
    legend=False,
    boxprops={"edgecolor": EDGE_COLOR},
    medianprops={"color": EDGE_COLOR, "linewidth": 1.4},
    whiskerprops={"color": EDGE_COLOR, "linewidth": 1.05},
    capprops={"color": EDGE_COLOR, "linewidth": 1.05},
    ax=ax
)

sns.stripplot(
    data=long_df,
    x="Model",
    y="Absolute error",
    hue="Model",
    palette=MODEL_PALETTE,
    size=2.8,
    alpha=0.52,
    jitter=0.15,
    edgecolor=EDGE_COLOR,
    linewidth=0.3,
    legend=False,
    ax=ax
)

ax.set_xlabel("")
ax.set_ylabel("Absolute error")
format_ax(ax, grid_axis="y")

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "30 Integrated M0 M3 comparison panel RMSE MAE only.png"),
    bbox_inches="tight"
)
plt.close()


# ---------------------------------------------------------
# 38 Observed、M0和M3箱线图，配对连线，均值和标准差，均值95%CI线
# ---------------------------------------------------------
value_wide = df[["Sample ID", "Actual", "M0", "M3"]].rename(
    columns={"Actual": "Observed"}
)

value_long = value_wide.melt(
    id_vars="Sample ID",
    value_vars=["Observed", "M0", "M3"],
    var_name="Series",
    value_name="Value"
)

series_order = ["Observed", "M0", "M3"]
x_pos = {name: i for i, name in enumerate(series_order)}

summary_records = []
for series in series_order:
    values = value_wide[series].values
    mean_value, sd_value, ci_low, ci_high = mean_ci95(values)
    summary_records.append({
        "Series": series,
        "x": x_pos[series],
        "Mean": mean_value,
        "SD": sd_value,
        "CI95 low": ci_low,
        "CI95 high": ci_high
    })

value_summary_df = pd.DataFrame(summary_records)
value_summary_df.to_excel(
    os.path.join(TABLE_DIR, "21 Observed M0 M3 均值 标准差 CI95.xlsx"),
    index=False
)

plt.figure(figsize=(7.2, 5.8))
ax = plt.gca()

for _, row in value_wide.iterrows():
    y_values = [row["Observed"], row["M0"], row["M3"]]
    ax.plot(
        [x_pos["Observed"], x_pos["M0"], x_pos["M3"]],
        y_values,
        color="#B8B8B8",
        linewidth=0.7,
        alpha=0.42,
        zorder=1
    )

sns.boxplot(
    data=value_long,
    x="Series",
    y="Value",
    order=series_order,
    palette=VALUE_LIGHT_PALETTE,
    width=0.42,
    linewidth=1.25,
    fliersize=0,
    boxprops={"edgecolor": EDGE_COLOR},
    medianprops={"color": EDGE_COLOR, "linewidth": 1.45},
    whiskerprops={"color": EDGE_COLOR, "linewidth": 1.10},
    capprops={"color": EDGE_COLOR, "linewidth": 1.10},
    ax=ax
)

sns.stripplot(
    data=value_long,
    x="Series",
    y="Value",
    order=series_order,
    palette=VALUE_PALETTE,
    size=4.0,
    jitter=0.13,
    alpha=0.72,
    edgecolor=EDGE_COLOR,
    linewidth=0.35,
    ax=ax
)

mean_x = value_summary_df["x"].values
mean_y = value_summary_df["Mean"].values
ci_low = value_summary_df["CI95 low"].values
ci_high = value_summary_df["CI95 high"].values

ax.plot(
    mean_x,
    mean_y,
    color="#1F1F1F",
    linewidth=2.2,
    marker="D",
    markersize=6.8,
    markerfacecolor="white",
    markeredgecolor="#1F1F1F",
    markeredgewidth=1.3,
    zorder=5,
    label="Mean"
)

ax.errorbar(
    mean_x,
    mean_y,
    yerr=[mean_y - ci_low, ci_high - mean_y],
    fmt="none",
    ecolor="#1F1F1F",
    elinewidth=1.4,
    capsize=5,
    capthick=1.3,
    zorder=4
)

y_span = value_long["Value"].max() - value_long["Value"].min()
if y_span <= 1e-12:
    y_span = 1.0

for _, row in value_summary_df.iterrows():
    ax.text(
        row["x"],
        row["Mean"] + y_span * 0.05,
        f"Mean={row['Mean']:.2f}\nSD={row['SD']:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            alpha=0.88
        ),
        zorder=6
    )

ax.set_xlabel("")
ax.set_ylabel("Value")
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR, loc="best")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "38 Observed M0 M3 boxline mean CI95.png"))


# =========================================================
# 16. 四等级商品等级预测能力绘图
# =========================================================

# 31 M0 四等级混淆矩阵
cm_m0 = grade_confusion_matrix_array(wide_diag["Observed grade"], wide_diag["M0 predicted grade"])

plt.figure(figsize=(6.4, 5.4))
ax = sns.heatmap(
    cm_m0,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=GRADE_LABELS,
    yticklabels=GRADE_LABELS,
    linewidths=0.85,
    linecolor="white",
    cbar_kws={"label": "Count"},
    annot_kws={"size": 15, "weight": "bold"}
)
ax.set_xlabel("")
ax.set_ylabel("Observed grade")
# 图顶部标题已按要求删除
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
save_fig(os.path.join(FIG_DIR, "31 Grade confusion matrix M0.png"))

# 32 M3 四等级混淆矩阵
cm_m3 = grade_confusion_matrix_array(wide_diag["Observed grade"], wide_diag["M3 predicted grade"])

plt.figure(figsize=(6.4, 5.4))
ax = sns.heatmap(
    cm_m3,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=GRADE_LABELS,
    yticklabels=GRADE_LABELS,
    linewidths=0.85,
    linecolor="white",
    cbar_kws={"label": "Count"},
    annot_kws={"size": 15, "weight": "bold"}
)
ax.set_xlabel("")
ax.set_ylabel("Observed grade")
# 图顶部标题已按要求删除
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
save_fig(os.path.join(FIG_DIR, "32 Grade confusion matrix M3.png"))

# 33 四等级总体分类指标
grade_main_metrics = [
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Cohen kappa"
]

grade_metric_long = grade_metric_df[["Model"] + grade_main_metrics].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(8.4, 5.2))
ax = sns.barplot(
    data=grade_metric_long,
    x="Metric",
    y="Value",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.58
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("Value")
# 图顶部标题已按要求删除

valid_values = grade_metric_long["Value"].replace([np.inf, -np.inf], np.nan).dropna().values
if len(valid_values) > 0:
    ymin = max(0, np.nanmin(valid_values) - 0.10)
    ymax = min(1.05, np.nanmax(valid_values) + 0.12)
    ax.set_ylim(ymin, ymax)

ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=22, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "33 Grade classification metrics without within one grade.png"))

# 34 四等级数量分布
grade_count_long = grade_count_df.melt(
    id_vars="Grade",
    value_vars=["Observed count", "M0 predicted count", "M3 predicted count"],
    var_name="Series",
    value_name="Count"
)

plt.figure(figsize=(8.6, 5.2))
ax = sns.barplot(
    data=grade_count_long,
    x="Grade",
    y="Count",
    hue="Series",
    palette=SERIES_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.05,
    width=0.66
)

shrink_bar_width(ax, width=0.18)
add_bar_labels(ax, fmt="%.0f", fontsize=12, padding=3)
set_count_ylim(ax, grade_count_long["Count"].values, top_margin=0.20)

ax.set_xlabel("")
ax.set_ylabel("Sample count")
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=18, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "34 Grade distribution.png"))

# 35 四等级逐等级 F1
plt.figure(figsize=(8.6, 5.2))
ax = sns.barplot(
    data=grade_report_df,
    x="Grade",
    y="F1",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.58
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("F1")
ax.set_ylim(0, 1.05)
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=18, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "35 Grade specific F1.png"))

# 36 四等级误差分布
grade_error_long = pd.concat([
    pd.DataFrame({
        "Model": "M0",
        "Grade error": wide_diag["M0 grade error"],
        "Absolute grade error": wide_diag["M0 absolute grade error"]
    }),
    pd.DataFrame({
        "Model": "M3",
        "Grade error": wide_diag["M3 grade error"],
        "Absolute grade error": wide_diag["M3 absolute grade error"]
    })
], axis=0, ignore_index=True)

plt.figure(figsize=(7.4, 5.2))
ax = sns.countplot(
    data=grade_error_long,
    x="Grade error",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.05,
    width=0.62
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.0f", fontsize=12, padding=3)

count_values = []
for c in ax.containers:
    for p in c:
        count_values.append(p.get_height())
set_count_ylim(ax, count_values, top_margin=0.20)

ax.axvline(x=0, color="#1F1F1F", linestyle="--", linewidth=1.05)
ax.set_xlabel("")
ax.set_ylabel("Sample count")
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "36 Grade error distribution.png"))

# 37 四等级逐等级 Accuracy
plt.figure(figsize=(8.6, 5.2))
ax = sns.barplot(
    data=grade_accuracy_df,
    x="Grade",
    y="Grade specific accuracy",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.58
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=18, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "37 Grade specific accuracy.png"))


# =========================================================
# 17. 三分类垩白风险等级绘图
# =========================================================

# 41 M0 三分类风险混淆矩阵
risk_cm_array_m0 = risk_confusion_matrix_array(wide_diag["Observed risk"], wide_diag["M0 predicted risk"])

plt.figure(figsize=(6.4, 5.2))
ax = sns.heatmap(
    risk_cm_array_m0,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=RISK_LABELS,
    yticklabels=RISK_LABELS,
    linewidths=0.85,
    linecolor="white",
    cbar_kws={"label": "Count"},
    annot_kws={"size": 15, "weight": "bold"}
)
ax.set_xlabel("")
ax.set_ylabel("Observed risk class")
# 图顶部标题已按要求删除
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
save_fig(os.path.join(FIG_DIR, "41 Risk confusion matrix M0.png"))

# 42 M3 三分类风险混淆矩阵
risk_cm_array_m3 = risk_confusion_matrix_array(wide_diag["Observed risk"], wide_diag["M3 predicted risk"])

plt.figure(figsize=(6.4, 5.2))
ax = sns.heatmap(
    risk_cm_array_m3,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=RISK_LABELS,
    yticklabels=RISK_LABELS,
    linewidths=0.85,
    linecolor="white",
    cbar_kws={"label": "Count"},
    annot_kws={"size": 15, "weight": "bold"}
)
ax.set_xlabel("")
ax.set_ylabel("Observed risk class")
# 图顶部标题已按要求删除
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
save_fig(os.path.join(FIG_DIR, "42 Risk confusion matrix M3.png"))

# 43 三分类风险总体分类指标
risk_main_metrics = [
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Cohen kappa"
]

risk_metric_long = risk_metric_df[["Model"] + risk_main_metrics].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(8.4, 5.2))
ax = sns.barplot(
    data=risk_metric_long,
    x="Metric",
    y="Value",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.58
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("Value")
# 图顶部标题已按要求删除

valid_values = risk_metric_long["Value"].replace([np.inf, -np.inf], np.nan).dropna().values
if len(valid_values) > 0:
    ymin = max(0, np.nanmin(valid_values) - 0.10)
    ymax = min(1.05, np.nanmax(valid_values) + 0.12)
    ax.set_ylim(ymin, ymax)

ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=22, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "43 Risk classification metrics.png"))

# 44 三分类风险逐等级 F1
plt.figure(figsize=(8.6, 5.2))
ax = sns.barplot(
    data=risk_report_df,
    x="Risk class",
    y="F1",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.58
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("F1")
ax.set_ylim(0, 1.05)
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=18, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "44 Risk specific F1.png"))

# 45 三分类风险逐等级 Accuracy
plt.figure(figsize=(8.6, 5.2))
ax = sns.barplot(
    data=risk_accuracy_df,
    x="Risk class",
    y="Risk specific accuracy",
    hue="Model",
    palette=MODEL_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.1,
    width=0.58
)

shrink_bar_width(ax, width=0.22)
add_bar_labels(ax, fmt="%.2f", fontsize=12, padding=3)

ax.set_xlabel("")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=18, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "45 Risk specific accuracy.png"))

# 46 三分类风险数量分布
risk_count_long = risk_count_df.melt(
    id_vars="Risk class",
    value_vars=["Observed count", "M0 predicted count", "M3 predicted count"],
    var_name="Series",
    value_name="Count"
)

plt.figure(figsize=(8.6, 5.2))
ax = sns.barplot(
    data=risk_count_long,
    x="Risk class",
    y="Count",
    hue="Series",
    palette=SERIES_PALETTE,
    edgecolor=EDGE_COLOR,
    linewidth=1.05,
    width=0.66
)

shrink_bar_width(ax, width=0.18)
add_bar_labels(ax, fmt="%.0f", fontsize=12, padding=3)
set_count_ylim(ax, risk_count_long["Count"].values, top_margin=0.20)

ax.set_xlabel("")
ax.set_ylabel("Sample count")
# 图顶部标题已按要求删除
ax.legend(frameon=True, edgecolor=EDGE_COLOR)
plt.xticks(rotation=18, ha="right")
format_ax(ax, grid_axis="y")
save_fig(os.path.join(FIG_DIR, "46 Risk distribution.png"))


# =========================================================
# 18. 输出简要结论文本
# =========================================================
m0_metrics = metrics_df[metrics_df["Model"] == "M0"].iloc[0]
m3_metrics = metrics_df[metrics_df["Model"] == "M3"].iloc[0]

m0_r2 = r2_df[r2_df["Model"] == "M0"]["R2"].iloc[0]
m3_r2 = r2_df[r2_df["Model"] == "M3"]["R2"].iloc[0]

m0_grade = grade_metric_df[grade_metric_df["Model"] == "M0"].iloc[0]
m3_grade = grade_metric_df[grade_metric_df["Model"] == "M3"].iloc[0]

m0_risk = risk_metric_df[risk_metric_df["Model"] == "M0"].iloc[0]
m3_risk = risk_metric_df[risk_metric_df["Model"] == "M3"].iloc[0]

rmse_change = improvement_df.loc[improvement_df["Metric"] == "RMSE", "Improvement percent"].values[0]
mae_change = improvement_df.loc[improvement_df["Metric"] == "MAE", "Improvement percent"].values[0]

grade_acc_change = grade_improvement_df.loc[
    grade_improvement_df["Metric"] == "Accuracy",
    "Improvement percent"
].values[0]

grade_kappa_change = grade_improvement_df.loc[
    grade_improvement_df["Metric"] == "Cohen kappa",
    "Improvement percent"
].values[0]

risk_acc_change = risk_improvement_df.loc[
    risk_improvement_df["Metric"] == "Accuracy",
    "Improvement percent"
].values[0]

risk_kappa_change = risk_improvement_df.loc[
    risk_improvement_df["Metric"] == "Cohen kappa",
    "Improvement percent"
].values[0]

better_rate = wide_diag["M3 better absolute error"].mean() * 100

summary_text = f"""
M0 与 M3 独立预测能力比较摘要

样本数：{len(df)}

一、连续数值预测能力，不含R2、MAPE和NRMSE

M0:
R2 = {m0_r2:.4f}
RMSE = {m0_metrics['RMSE']:.4f}
MAE = {m0_metrics['MAE']:.4f}
Lin CCC = {m0_metrics['Lin CCC']:.4f}
Willmott d = {m0_metrics['Willmott d']:.4f}
RPD = {m0_metrics['RPD']:.4f}
Pearson r = {m0_metrics['Pearson r']:.4f}

M3:
R2 = {m3_r2:.4f}
RMSE = {m3_metrics['RMSE']:.4f}
MAE = {m3_metrics['MAE']:.4f}
Lin CCC = {m3_metrics['Lin CCC']:.4f}
Willmott d = {m3_metrics['Willmott d']:.4f}
RPD = {m3_metrics['RPD']:.4f}
Pearson r = {m3_metrics['Pearson r']:.4f}

M3相对M0:
RMSE降低率 = {rmse_change:.2f}%
MAE降低率 = {mae_change:.2f}%
M3绝对误差小于M0的样本比例 = {better_rate:.2f}%

二、四等级商品等级预测能力

四等级划分：
≤2 为 Grade 1
2到5 为 Grade 2
5到8 为 Grade 3
>8 为 Out of grade

M0:
Accuracy = {m0_grade['Accuracy']:.4f}
Macro F1 = {m0_grade['Macro F1']:.4f}
Weighted F1 = {m0_grade['Weighted F1']:.4f}
Cohen kappa = {m0_grade['Cohen kappa']:.4f}
Within one grade accuracy = {m0_grade['Within one grade accuracy']:.4f}
Mean absolute grade error = {m0_grade['Mean absolute grade error']:.4f}

M3:
Accuracy = {m3_grade['Accuracy']:.4f}
Macro F1 = {m3_grade['Macro F1']:.4f}
Weighted F1 = {m3_grade['Weighted F1']:.4f}
Cohen kappa = {m3_grade['Cohen kappa']:.4f}
Within one grade accuracy = {m3_grade['Within one grade accuracy']:.4f}
Mean absolute grade error = {m3_grade['Mean absolute grade error']:.4f}

M3相对M0:
四等级Accuracy改进率 = {grade_acc_change:.2f}%
四等级Cohen kappa改进率 = {grade_kappa_change:.2f}%

三、三分类垩白风险等级预测能力

三分类风险等级划分：
≤5 为 Low chalkiness risk
5到8 为 Moderate chalkiness risk
>8 为 High chalkiness risk

M0:
Accuracy = {m0_risk['Accuracy']:.4f}
Macro F1 = {m0_risk['Macro F1']:.4f}
Weighted F1 = {m0_risk['Weighted F1']:.4f}
Cohen kappa = {m0_risk['Cohen kappa']:.4f}
Mean absolute risk error = {m0_risk['Mean absolute risk error']:.4f}

M3:
Accuracy = {m3_risk['Accuracy']:.4f}
Macro F1 = {m3_risk['Macro F1']:.4f}
Weighted F1 = {m3_risk['Weighted F1']:.4f}
Cohen kappa = {m3_risk['Cohen kappa']:.4f}
Mean absolute risk error = {m3_risk['Mean absolute risk error']:.4f}

M3相对M0:
三分类风险Accuracy改进率 = {risk_acc_change:.2f}%
三分类风险Cohen kappa改进率 = {risk_kappa_change:.2f}%

连续预测图件：
00 Prediction R2 M0 M3.png
01 Observed vs predicted M0 M3.png
07 Residual density.png
09 Absolute error boxplot.png
21 Absolute error by observed value group.png
30 Integrated M0 M3 comparison panel RMSE MAE only.png
38 Observed M0 M3 boxline mean CI95.png

四等级商品等级图件：
31 Grade confusion matrix M0.png
32 Grade confusion matrix M3.png
33 Grade classification metrics without within one grade.png
34 Grade distribution.png
35 Grade specific F1.png
36 Grade error distribution.png
37 Grade specific accuracy.png

三分类垩白风险等级图件：
41 Risk confusion matrix M0.png
42 Risk confusion matrix M3.png
43 Risk classification metrics.png
44 Risk specific F1.png
45 Risk specific accuracy.png
46 Risk distribution.png
"""

with open(
    os.path.join(OUTPUT_DIR, "M0 M3预测能力比较摘要 含风险等级 箱线配对图 空格指标名.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(summary_text)

print(summary_text)

print("全部结果已输出：")
print(f"总目录：{OUTPUT_DIR}")
print(f"表格目录：{TABLE_DIR}")
print(f"图件目录：{FIG_DIR}")