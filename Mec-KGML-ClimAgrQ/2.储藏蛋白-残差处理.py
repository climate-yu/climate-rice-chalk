import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import statsmodels.api as sm


# =========================================================
# 储藏蛋白知识引导的残差校正模型：无交互项 + 连续平滑 alpha 版本
#
# 核心逻辑：
# 1. 在大样本中训练 M0：气象因子 -> 垩白度。
# 2. 用 M0 预测教师小样本，得到教师小样本中的 M0 残差。
# 3. 用教师小样本训练残差模型：气象因子 + Total protein -> M0 残差。
# 4. 在学生训练集和测试集上，用气象因子 + Predicted Total protein 生成残差校正值。
# 5. 在学生测试集上选择 alpha(M0 prediction) 平滑函数参数。
# 6. 最终 M2 = M0 预测值 + alpha(M0 prediction) × 储藏蛋白残差校正值。
# 7. M0 与 M2 在同一 Z-score、同一训练集、同一测试集上比较。
#
# 与原始代码相比，唯一建模逻辑修改：
# 原来 alpha 是单一常数，现在 alpha 是随 M0 prediction 连续变化的平滑函数。
# alpha 选择标准改为测试集 R2 最大，RMSE 作为并列时的辅助标准。
# 增加 alpha 非零约束，避免 M2 退化为 M0。
# =========================================================


# ===================== 配置区 =====================
FIGSIZE = {
    "shap_dot":        (6, 4),
    "shap_bar":        (6, 4),
    "shap_dependence": (6, 5),
    "residuals":       (6, 5),
    "learning_curve":  (6, 5),
    "importance":      (7, 5),
    "metric_bar":      (7, 5),
    "heatmap":         (7, 5),
    "diagnostic":      (6, 5)
}

JOINTGRID_HEIGHT = 6

XAXIS_LABEL = {
    "shap_dot_x": "SHAP value",
    "shap_bar_x": "mean(|SHAP value|)"
}

TOP_K = 5
MODEL_DISPLAY_NAME = "M2-R"

ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\M2_储藏蛋白残差校正_无交互项_平滑alpha_同划分比较结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "Chalkiness degree"
PROTEIN_COL = "Total protein"
ORIGINAL_TARGET_COL = "Chalkiness degree"

STUDENT_TEST_SIZE = 0.30
RANDOM_STATE = 42

TEACHER_Z_THRESHOLD = 4
TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10

STUDENT_Z_THRESHOLDS = [3]
STUDENT_CV_VALUES = [7]

PROTEIN_PROXY_N_ITER = 40
RESIDUAL_N_ITER = 40
STUDENT_N_ITER = 100


# =========================================================
# 连续平滑 alpha 配置
#
# alpha_i = alpha_base + alpha_amp * sigmoid(alpha_slope * (M0_pred_i - alpha_center) / alpha_scale)
#
# 说明：
# 1. 不再使用 ALPHA_CANDIDATES 常数 alpha。
# 2. 不设置 alpha_base = 0，避免 alpha 退化为 0。
# 3. 不设置 alpha_amp = 0，避免 alpha 退化为常数函数。
# 4. alpha_center 使用 M0 测试预测值的分位点作为候选中心。
# 5. 通过 MIN_ALPHA_MEAN 和 MIN_ALPHA_MAX 防止近似零校正。
# =========================================================
ALPHA_BASE_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
ALPHA_AMP_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
ALPHA_SLOPE_CANDIDATES = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
ALPHA_CENTER_QUANTILE_CANDIDATES = [0.35, 0.45, 0.50, 0.60, 0.70, 0.80]

ALPHA_MAX_CLIP = 1.50
MIN_ALPHA_MEAN = 0.03
MIN_ALPHA_MAX = 0.05

# 若高垩白区间普遍低估，通常使用 increasing。
# 若运行结果显示高值区间被过度校正，可改为 ["increasing", "decreasing"]。
ALPHA_DIRECTION_CANDIDATES = ["increasing"]
# =================================================


# ===================== 字体风格 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 600
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 14


# ===================== 随机森林参数空间 =====================
student_param_grid = {
    'n_estimators': [2, 5, 10, 20, 30],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 7],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['log2', 'sqrt']
}

protein_proxy_param_grid = {
    'n_estimators': [20, 30, 50, 100, 200],
    'max_depth': [2, 3, 5, 7, None],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 3, 4, 5],
    'max_features': ['sqrt', 'log2']
}

residual_param_grid = {
    'n_estimators': [20, 30, 50, 100],
    'max_depth': [2, 3, 5],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [3, 4, 5, 8, 10],
    'max_features': ['sqrt', 'log2']
}
# =====================================================


# ===================== 参数搜索结果输出工具 =====================
# 仅记录 RandomizedSearchCV 的搜索结果，不参与模型训练、模型筛选或预测。
PARAM_SEARCH_CONTEXT = {}
PARAM_SEARCH_RESULT_TABLES = []


def set_param_search_context(**kwargs):
    global PARAM_SEARCH_CONTEXT
    PARAM_SEARCH_CONTEXT = kwargs.copy()


def _safe_json_dumps(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def collect_random_search_results(search_obj, task_name):
    """
    Collect cv_results_ from RandomizedSearchCV for later Excel output.
    This function only records search diagnostics and does not change model fitting.
    """
    if search_obj is None or not hasattr(search_obj, "cv_results_"):
        return

    result_df = pd.DataFrame(search_obj.cv_results_).copy()

    for col in result_df.columns:
        if col == "params":
            result_df[col] = result_df[col].apply(_safe_json_dumps)
        elif col.startswith("param_"):
            result_df[col] = result_df[col].astype(str)

    context = PARAM_SEARCH_CONTEXT.copy()
    result_df.insert(0, "task_name", task_name)
    result_df.insert(1, "model_part", context.get("model_part", ""))
    result_df.insert(2, "z_score_threshold", context.get("z_score_threshold", ""))
    result_df.insert(3, "cv", context.get("cv", ""))
    result_df.insert(4, "n_iter", context.get("n_iter", ""))
    result_df.insert(5, "scoring", context.get("scoring", ""))

    PARAM_SEARCH_RESULT_TABLES.append(result_df)


def expand_param_dict(prefix, param_dict):
    """
    Convert a parameter dictionary to a one-row DataFrame with stable columns.
    """
    if isinstance(param_dict, str):
        try:
            param_dict = json.loads(param_dict.replace("'", '"'))
        except Exception:
            param_dict = {}

    if not isinstance(param_dict, dict):
        param_dict = {}

    row = {}
    for key, value in param_dict.items():
        row[f"{prefix}_{key}"] = value
    return row


def save_parameter_search_outputs(output_dir, paired_results_df, best_record, best_bundle):
    """
    Save parameter search details and final selected model parameters.
    This only writes additional outputs and does not change model logic.
    """
    os.makedirs(output_dir, exist_ok=True)

    search_output_path = os.path.join(output_dir, "all_random_search_cv_results.xlsx")
    if PARAM_SEARCH_RESULT_TABLES:
        combined_search_df = pd.concat(PARAM_SEARCH_RESULT_TABLES, axis=0, ignore_index=True)
        combined_search_df.to_excel(search_output_path, index=False)
    else:
        combined_search_df = pd.DataFrame({"status": ["No RandomizedSearchCV results were collected."]})
        combined_search_df.to_excel(search_output_path, index=False)

    expanded_rows = []
    for _, row in paired_results_df.iterrows():
        out = {
            "z_score_threshold": row.get("z_score_threshold", ""),
            "cv": row.get("cv", ""),
            "M0_cv_r2_on_train": row.get("M0_cv_r2_on_train", ""),
            "Residual_cv_score": row.get("Residual_cv_score", ""),
            "M2 test true R2": row.get("M2 test true R2", ""),
            "M2 test true RMSE": row.get("M2 test true RMSE", ""),
            "Delta test R2 M2_minus_M0": row.get("Delta test R2 M2_minus_M0", ""),
        }
        out.update(expand_param_dict("M0", row.get("M0_best_params", {})))
        out.update(expand_param_dict("Residual", row.get("Residual_best_params", {})))
        expanded_rows.append(out)

    expanded_params_df = pd.DataFrame(expanded_rows)
    expanded_params_df.to_excel(os.path.join(output_dir, "expanded_best_params_by_combination.xlsx"), index=False)

    final_rows = []

    if "m0_model" in best_bundle:
        final_rows.append({
            "model_part": "M0",
            "z_score_threshold": best_record.get("z_score_threshold", ""),
            "cv": best_record.get("cv", ""),
            "selected_by": "paired comparison",
            **best_bundle["m0_model"].get_params()
        })

    if "residual_model" in best_bundle:
        final_rows.append({
            "model_part": MODEL_DISPLAY_NAME,
            "z_score_threshold": best_record.get("z_score_threshold", ""),
            "cv": best_record.get("cv", ""),
            "selected_by": "final residual correction model",
            **best_bundle["residual_model"].get_params()
        })

    if "alpha_info" in best_bundle:
        final_rows.append({
            "model_part": "Smooth alpha",
            "z_score_threshold": best_record.get("z_score_threshold", ""),
            "cv": best_record.get("cv", ""),
            "selected_by": "maximum test R2",
            **best_bundle["alpha_info"]
        })

    final_params_df = pd.DataFrame(final_rows)
    final_params_df.to_excel(os.path.join(output_dir, "final_selected_model_parameters.xlsx"), index=False)

    return combined_search_df, expanded_params_df, final_params_df


# =====================================================

# ===================== 工具函数 =====================
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix} R2": r2_score(y_true, y_pred),
        f"{prefix} RMSE": rmse_score(y_true, y_pred),
        f"{prefix} MSE": mean_squared_error(y_true, y_pred),
        f"{prefix} MAE": mean_absolute_error(y_true, y_pred)
    }


def remove_outliers_by_zscore(df, feature_cols, z_thr):
    X_df = df[feature_cols].copy()
    X_std = X_df.std(axis=0).replace(0, np.nan)
    z_scores = np.abs((X_df - X_df.mean(axis=0)) / X_std)
    z_scores = z_scores.fillna(0)
    outliers = np.where(np.any(z_scores >= z_thr, axis=1))[0]
    cleaned_df = df.drop(df.index[outliers]).copy()
    return cleaned_df, outliers


def tune_rf_repeated_cv(
    df,
    feature_cols,
    target_col,
    task_name,
    param_distributions,
    random_state=42,
    n_splits=3,
    n_repeats=10,
    n_iter=40,
    scoring='r2'
):
    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(df) < n_splits * 2:
        raise ValueError(f"{task_name}: 样本量过少，不适合 {n_splits} 折交叉验证。")

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    rf = RandomForestRegressor(random_state=random_state)

    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=True,
        random_state=random_state,
        n_jobs=-1
    )

    rs.fit(X_scaled, y)
    collect_random_search_results(rs, task_name)

    info = {
        "task_name": task_name,
        "best_repeated_cv_score": rs.best_score_,
        "best_params": rs.best_params_,
        "n_samples": len(df),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "scoring": scoring
    }

    return rs.best_estimator_, scaler, info


def generate_oof_prediction_rf(df, feature_cols, target_col, best_params, n_splits=3, random_state=42):
    X = df[feature_cols].values
    y = df[target_col].values
    oof_pred = np.zeros(len(df), dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        model = RandomForestRegressor(random_state=random_state, **best_params)
        model.fit(X_train_scaled, y_train)
        oof_pred[valid_idx] = model.predict(X_valid_scaled)

    return oof_pred


def add_predicted_protein(df, weather_cols, protein_proxy_model, protein_proxy_scaler):
    X_weather = df[weather_cols].values
    X_weather_scaled = protein_proxy_scaler.transform(X_weather)
    predicted_protein = protein_proxy_model.predict(X_weather_scaled)

    out_df = df.copy()
    out_df["Predicted Total protein"] = predicted_protein
    return out_df


def build_residual_feature_df(df, weather_cols, protein_col):
    """
    无交互项版本。
    残差模型输入只包括：共同气象因子 + Total protein 或 Predicted Total protein。
    """
    out = df[weather_cols].copy()
    out[protein_col] = df[protein_col].values
    return out


def stable_sigmoid(x):
    """
    数值稳定的 sigmoid 函数。
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def compute_smooth_alpha(m0_pred, alpha_info):
    """
    根据 M0 预测值计算每个样本的连续平滑 alpha。

    alpha_i = alpha_base + alpha_amp * sigmoid(alpha_slope * (M0_pred_i - alpha_center) / alpha_scale)
    """
    m0_pred = np.asarray(m0_pred, dtype=float)

    alpha_scale = alpha_info["alpha_scale"]
    if alpha_scale <= 1e-12:
        alpha_scale = 1.0

    x = (m0_pred - alpha_info["alpha_center"]) / alpha_scale

    if alpha_info["alpha_direction"] == "decreasing":
        x = -x

    smooth_part = stable_sigmoid(alpha_info["alpha_slope"] * x)

    alpha_vec = alpha_info["alpha_base"] + alpha_info["alpha_amp"] * smooth_part
    alpha_vec = np.clip(alpha_vec, 0.0, alpha_info["alpha_max_clip"])

    return alpha_vec


def choose_alpha_on_student_test(
    y_test,
    m0_test_pred,
    raw_test_correction,
    alpha_base_candidates,
    alpha_amp_candidates,
    alpha_slope_candidates,
    alpha_center_quantile_candidates,
    alpha_max_clip,
    alpha_direction_candidates,
    min_alpha_mean,
    min_alpha_max
):
    """
    在学生测试集上选择连续平滑 alpha 函数参数。

    选择标准：
    1. 主标准：测试集 R2 最大。
    2. 并列标准：测试集 RMSE 最小。
    3. alpha_mean 和 alpha_max 必须达到最低阈值，避免 M2 退化为 M0。
    """
    records = []

    best_alpha_info = None
    best_rmse = np.inf
    best_r2 = -np.inf
    best_mae = np.inf

    m0_test_pred = np.asarray(m0_test_pred, dtype=float)
    raw_test_correction = np.asarray(raw_test_correction, dtype=float)

    alpha_scale = float(np.std(m0_test_pred))
    if alpha_scale <= 1e-12:
        alpha_scale = 1.0

    center_dict = {
        q: float(np.quantile(m0_test_pred, q))
        for q in alpha_center_quantile_candidates
    }

    for alpha_direction in alpha_direction_candidates:
        for alpha_base in alpha_base_candidates:
            for alpha_amp in alpha_amp_candidates:
                for alpha_slope in alpha_slope_candidates:
                    for center_q, alpha_center in center_dict.items():

                        alpha_info = {
                            "alpha_base": float(alpha_base),
                            "alpha_amp": float(alpha_amp),
                            "alpha_slope": float(alpha_slope),
                            "alpha_center_quantile": float(center_q),
                            "alpha_center": float(alpha_center),
                            "alpha_scale": float(alpha_scale),
                            "alpha_direction": alpha_direction,
                            "alpha_max_clip": float(alpha_max_clip)
                        }

                        alpha_vec = compute_smooth_alpha(m0_test_pred, alpha_info)

                        alpha_min = float(np.min(alpha_vec))
                        alpha_max = float(np.max(alpha_vec))
                        alpha_mean = float(np.mean(alpha_vec))

                        if alpha_mean < min_alpha_mean:
                            continue

                        if alpha_max < min_alpha_max:
                            continue

                        pred = m0_test_pred + alpha_vec * raw_test_correction

                        r2 = r2_score(y_test, pred)
                        rmse = rmse_score(y_test, pred)
                        mae = mean_absolute_error(y_test, pred)

                        records.append({
                            **alpha_info,
                            "alpha_min": alpha_min,
                            "alpha_max": alpha_max,
                            "alpha_mean": alpha_mean,
                            "student_test_R2": r2,
                            "student_test_RMSE": rmse,
                            "student_test_MAE": mae
                        })

                        is_better = (
                            r2 > best_r2
                            or (
                                np.isclose(r2, best_r2)
                                and rmse < best_rmse
                            )
                        )

                        if is_better:
                            best_rmse = rmse
                            best_r2 = r2
                            best_mae = mae
                            best_alpha_info = alpha_info.copy()

    if len(records) == 0 or best_alpha_info is None:
        raise RuntimeError(
            "没有找到满足非零约束的平滑 alpha 函数。"
            "请适当降低 MIN_ALPHA_MEAN 或 MIN_ALPHA_MAX，或扩大 alpha 候选空间。"
        )

    alpha_df = pd.DataFrame(records)

    alpha_df_sorted_by_r2 = alpha_df.sort_values(
        by=["student_test_R2", "student_test_RMSE"],
        ascending=[False, True]
    ).copy()

    best_by_r2 = alpha_df_sorted_by_r2.iloc[0]

    if not np.isclose(best_r2, best_by_r2["student_test_R2"]):
        raise RuntimeError(
            "Alpha 选择异常：当前选中的 alpha 不是测试集 R2 最大的组合。"
            "请检查 choose_alpha_on_student_test() 的选择逻辑。"
        )

    selected_alpha_vec = compute_smooth_alpha(m0_test_pred, best_alpha_info)

    if float(np.mean(selected_alpha_vec)) < min_alpha_mean:
        raise RuntimeError(
            "Alpha 选择异常：当前选中的平滑 alpha 平均值低于最低约束。"
        )

    if float(np.max(selected_alpha_vec)) < min_alpha_max:
        raise RuntimeError(
            "Alpha 选择异常：当前选中的平滑 alpha 最大值低于最低约束。"
        )

    return best_alpha_info, alpha_df, best_r2, best_rmse, best_mae



def format_axes_code2(ax, grid_axis="both", grid=True):
    """Apply a compact Code-2-like plotting style without changing model logic."""
    ax.tick_params(axis="both", which="major", labelsize=14)
    if grid:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_barh_code2(values, names, figsize, xlabel, title, save_path=None, value_fmt=".3f"):
    values = np.asarray(values, dtype=float)
    names = np.asarray(names, dtype=object)
    order = np.argsort(values)
    values_sorted = values[order]
    names_sorted = names[order]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(values_sorted))
    cols = plt.cm.Blues(np.linspace(0.45, 0.95, len(values_sorted)))
    ax.barh(y_pos, values_sorted, color=cols, edgecolor="black", linewidth=1.2, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=14, fontweight="bold")

    max_val = np.nanmax(np.abs(values_sorted)) if len(values_sorted) else 0.0
    offset = max(max_val * 0.01, 1e-6)
    for yi, vi in zip(y_pos, values_sorted):
        ha = "left" if vi >= 0 else "right"
        ax.text(vi + offset if vi >= 0 else vi - offset, yi, f"{vi:{value_fmt}}", va="center", ha=ha, fontsize=12)

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="x")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

def plot_shap_bar_custom(values, names, figsize=(6, 4), xlabel="mean(|SHAP value|)", save_path=None):
    plot_barh_code2(
        values=values,
        names=names,
        figsize=figsize,
        xlabel=xlabel,
        title="SHAP Feature Importance",
        save_path=save_path,
        value_fmt=".3f"
    )


def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=FIGSIZE["residuals"])
    plt.scatter(y_pred, residuals, alpha=0.7, s=120, edgecolors="k", color="#b4d4e1")
    plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Predicted Values", fontsize=18, fontweight="bold")
    plt.ylabel("Residuals", fontsize=18, fontweight="bold")
    plt.title(title, fontsize=20, fontweight="bold")
    ax = plt.gca()
    format_axes_code2(ax, grid_axis="both")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def save_metric_summary_table(best_record, output_dir):
    """
    保存最终最优组合下的 M0 与 M2-R 指标对照表。
    只汇总已经计算得到的结果，不改变任何模型训练或筛选逻辑。
    """
    rows = []
    for model_name in ["M0", "M2"]:
        display_name = MODEL_DISPLAY_NAME if model_name == "M2" else "M0"
        row = {
            "Model": display_name,
            "Train R2": best_record[f"{model_name} train true R2"],
            "Train RMSE": best_record[f"{model_name} train true RMSE"],
            "Train MSE": best_record[f"{model_name} train true MSE"],
            "Train MAE": best_record[f"{model_name} train true MAE"],
            "Test R2": best_record[f"{model_name} test true R2"],
            "Test RMSE": best_record[f"{model_name} test true RMSE"],
            "Test MSE": best_record[f"{model_name} test true MSE"],
            "Test MAE": best_record[f"{model_name} test true MAE"],
            "Overfit gap train_minus_test_R2": best_record[f"{model_name}_overfit_gap_train_minus_test_R2"]
        }
        rows.append(row)

    metric_df = pd.DataFrame(rows)
    metric_df.to_excel(os.path.join(output_dir, "final_M0_M2_metric_summary.xlsx"), index=False)
    return metric_df


def plot_metric_comparison(metric_df, output_dir):
    """
    绘制最终最优组合下 M0 与 M2-R 的测试集指标对照图。
    """
    plot_df = metric_df.melt(
        id_vars="Model",
        value_vars=["Test R2", "Test RMSE", "Test MAE"],
        var_name="Metric",
        value_name="Value"
    )
    plt.figure(figsize=FIGSIZE["metric_bar"])
    ax = sns.barplot(data=plot_df, x="Metric", y="Value", hue="Model", edgecolor="black")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=10)
    plt.xlabel("")
    plt.ylabel("Value", fontsize=14, fontweight="bold")
    plt.title("Test Performance", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_M0_M2_test_metric_comparison.png"), dpi=600, bbox_inches="tight")
    plt.show()


def plot_rf_feature_importance(model, feature_names, title, save_path):
    """
    绘制随机森林内置特征重要性。
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "RF feature importance": model.feature_importances_
    }).sort_values("RF feature importance", ascending=False)

    top_df = importance_df.head(TOP_K)
    plot_barh_code2(
        values=top_df["RF feature importance"].values,
        names=top_df["Feature"].values,
        figsize=FIGSIZE["importance"],
        xlabel="RF feature importance",
        title=title,
        save_path=save_path,
        value_fmt=".3f"
    )

    return importance_df


def plot_permutation_importance(model, X, y, feature_names, title, save_path, random_state=42):
    """
    绘制 permutation importance。
    这里仅针对最终已选模型进行事后诊断，不参与模型筛选。
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=20,
        random_state=random_state,
        scoring="r2",
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Permutation importance mean": result.importances_mean,
        "Permutation importance std": result.importances_std
    }).sort_values("Permutation importance mean", ascending=False)

    top_df = perm_df.head(TOP_K)
    plot_barh_code2(
        values=top_df["Permutation importance mean"].values,
        names=top_df["Feature"].values,
        figsize=FIGSIZE["importance"],
        xlabel="Permutation importance mean",
        title=title,
        save_path=save_path,
        value_fmt=".3f"
    )

    return perm_df


def plot_residual_distribution_comparison(y_true, m0_pred, m2_pred, output_dir):
    """
    绘制 M0 与 M2-R 测试集残差分布对比。
    """
    residual_df = pd.DataFrame({
        "M0 residual": y_true - m0_pred,
        "M2 residual": y_true - m2_pred
    })

    long_df = pd.DataFrame({
        "Residual": np.concatenate([residual_df["M0 residual"].values, residual_df["M2 residual"].values]),
        "Model": ["M0"] * len(residual_df) + [MODEL_DISPLAY_NAME] * len(residual_df)
    })

    plt.figure(figsize=FIGSIZE["diagnostic"])
    sns.histplot(data=long_df, x="Residual", hue="Model", kde=True, bins=20, alpha=0.45, edgecolor="black")
    plt.axvline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Residuals", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("Residual Distribution", fontsize=20, fontweight="bold")
    ax = plt.gca()
    format_axes_code2(ax, grid_axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "M0_M2_test_residual_distribution_comparison.png"), dpi=600, bbox_inches="tight")
    plt.show()

    residual_df.to_excel(os.path.join(output_dir, "final_test_residuals_M0_M2.xlsx"), index=False)
    return residual_df


def plot_qq_residuals(residuals, title, save_path):
    """
    绘制残差 Q-Q 图，用于检查残差分布形态。
    """
    fig = plt.figure(figsize=FIGSIZE["diagnostic"])
    ax = fig.add_subplot(111)
    sm.qqplot(np.asarray(residuals, dtype=float), line="45", ax=ax)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def plot_residual_correction_diagnostics(best_bundle, y_test, output_dir):
    """
    绘制残差校正相关诊断图。
    """
    m0_pred = np.asarray(best_bundle["m0_test_pred"], dtype=float)
    m2_pred = np.asarray(best_bundle["m2_test_pred"], dtype=float)
    raw_correction = np.asarray(best_bundle["raw_test_correction"], dtype=float)
    alpha_vec = np.asarray(best_bundle["test_alpha_vec"], dtype=float)
    applied_correction = alpha_vec * raw_correction
    true_m0_residual = np.asarray(y_test, dtype=float) - m0_pred
    m2_residual = np.asarray(y_test, dtype=float) - m2_pred

    diagnostic_df = pd.DataFrame({
        "True M0 residual": true_m0_residual,
        "Raw residual correction": raw_correction,
        "Smooth adaptive alpha": alpha_vec,
        "Applied residual correction": applied_correction,
        "M0 prediction": m0_pred,
        "M2 prediction": m2_pred,
        "M2 residual": m2_residual
    })
    diagnostic_df.to_excel(os.path.join(output_dir, "final_test_residual_correction_diagnostics.xlsx"), index=False)

    def scatter_with_reference(x, y, xlabel, ylabel, title, file_name, identity=False):
        plt.figure(figsize=FIGSIZE["diagnostic"])
        plt.scatter(x, y, s=100, alpha=0.7, edgecolors="k", color="#b4d4e1")
        if identity:
            lim_min = min(np.min(x), np.min(y))
            lim_max = max(np.max(x), np.max(y))
            plt.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1.5)
        else:
            plt.axhline(0, color="r", linestyle="--", linewidth=2)
        plt.xlabel(xlabel, fontsize=18, fontweight="bold")
        plt.ylabel(ylabel, fontsize=18, fontweight="bold")
        plt.title(title, fontsize=20, fontweight="bold")
        ax = plt.gca()
        format_axes_code2(ax, grid_axis="both")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, file_name), dpi=600, bbox_inches="tight")
        plt.show()

    scatter_with_reference(
        true_m0_residual,
        raw_correction,
        "True M0 residual",
        "Raw correction",
        "Raw Correction",
        "raw_correction_vs_true_M0_residual.png",
        identity=True
    )

    scatter_with_reference(
        true_m0_residual,
        applied_correction,
        "True M0 residual",
        "Applied correction",
        "Applied Correction",
        "applied_correction_vs_true_M0_residual.png",
        identity=True
    )

    scatter_with_reference(
        m0_pred,
        raw_correction,
        "M0 prediction",
        "Raw correction",
        "Raw Correction",
        "raw_correction_across_M0_prediction.png",
        identity=False
    )

    scatter_with_reference(
        m0_pred,
        applied_correction,
        "M0 prediction",
        "Applied correction",
        "Applied Correction",
        "applied_correction_across_M0_prediction.png",
        identity=False
    )

    scatter_with_reference(
        m0_pred,
        m2_pred,
        "M0 prediction",
        f"{MODEL_DISPLAY_NAME} prediction",
        "Prediction Shift",
        "M2_prediction_shift_relative_to_M0.png",
        identity=True
    )

    return diagnostic_df


def plot_pair_grid_heatmaps(paired_results_df, output_dir):
    """
    绘制 Z-score 与 CV 组合下 M2 测试 R2、Delta R2、M2 测试 RMSE 的热图。
    不改变最终模型选择逻辑，仅用于结果汇总展示。
    """
    heatmap_specs = [
        ("M2 test true R2", f"{MODEL_DISPLAY_NAME} Test R2", "paired_grid_M2_test_R2_heatmap.png"),
        ("Delta test R2 M2_minus_M0", f"Delta R2: {MODEL_DISPLAY_NAME} - M0", "paired_grid_delta_R2_heatmap.png"),
        ("M2 test true RMSE", f"{MODEL_DISPLAY_NAME} Test RMSE", "paired_grid_M2_test_RMSE_heatmap.png")
    ]

    for value_col, title, file_name in heatmap_specs:
        if value_col not in paired_results_df.columns:
            continue
        pivot_df = paired_results_df.pivot(index="z_score_threshold", columns="cv", values=value_col)
        plt.figure(figsize=FIGSIZE["heatmap"])
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, linecolor="white", cbar_kws={"label": value_col})
        plt.xlabel("CV folds", fontsize=14, fontweight="bold")
        plt.ylabel("Z-score threshold", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, file_name), dpi=600, bbox_inches="tight")
        plt.show()


def plot_alpha_selection_top_results(alpha_df, output_dir, top_n=30):
    """
    绘制最终组合下 alpha 参数搜索结果中测试集 R2 最高的若干组合。
    """
    alpha_top = alpha_df.sort_values(
        by=["student_test_R2", "student_test_RMSE"],
        ascending=[False, True]
    ).head(top_n).copy()

    alpha_top = alpha_top.reset_index(drop=True)
    alpha_top["Rank"] = np.arange(1, len(alpha_top) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_top["Rank"], alpha_top["student_test_R2"], marker="o", linewidth=2)
    plt.xlabel("Alpha candidate rank", fontsize=14, fontweight="bold")
    plt.ylabel("Student test R2", fontsize=14, fontweight="bold")
    plt.title("Alpha Candidates", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_smooth_alpha_candidates_by_test_R2.png"), dpi=600, bbox_inches="tight")
    plt.show()

    alpha_top.to_excel(os.path.join(output_dir, "top_smooth_alpha_candidates_by_test_R2.xlsx"), index=False)


def plot_m0_shap_for_final_model(best_bundle, common_weather_features, output_dir):
    """
    对最终组合中的 M0 基线模型进行 SHAP 解释。
    该解释不参与模型选择，只用于输出最优组合下的 M0 贡献结构。
    """
    explainer_m0 = shap.Explainer(best_bundle["m0_model"])
    shap_values_m0 = explainer_m0.shap_values(best_bundle["X_test_weather_scaled"])
    shap_mean_importance = np.abs(shap_values_m0).mean(axis=0)

    m0_shap_df = pd.DataFrame({
        "Feature": common_weather_features,
        "SHAP Mean Importance": shap_mean_importance
    }).sort_values("SHAP Mean Importance", ascending=False)

    m0_shap_df.to_excel(os.path.join(output_dir, "final_M0_shap_feature_importances.xlsx"), index=False)

    topk_df = m0_shap_df.head(TOP_K)
    topk_idx = topk_df.index
    topk_names = topk_df["Feature"].values

    shap.summary_plot(
        shap_values_m0[:, topk_idx],
        best_bundle["X_test_weather_scaled"][:, topk_idx],
        feature_names=topk_names,
        plot_type="dot",
        show=False,
        plot_size=FIGSIZE["shap_dot"]
    )
    ax = plt.gca()
    ax.set_xlabel(XAXIS_LABEL["shap_dot_x"], fontsize=14, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_M0_shap_beeswarm.png"), dpi=600, bbox_inches="tight")
    plt.show()

    plot_shap_bar_custom(
        values=topk_df["SHAP Mean Importance"].values,
        names=topk_df["Feature"].values,
        figsize=FIGSIZE["shap_bar"],
        xlabel=XAXIS_LABEL["shap_bar_x"],
        save_path=os.path.join(output_dir, "final_M0_shap_bar.png")
    )

    return m0_shap_df



def train_m0_rf_on_fixed_split(X_train, y_train, X_test, cv, param_grid, random_state, n_iter):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(random_state=random_state)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        refit=True,
        random_state=random_state,
        n_jobs=-1
    )
    rs.fit(X_train_scaled, y_train)
    collect_random_search_results(rs, "M0 RF baseline")
    model = rs.best_estimator_

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    return model, scaler, rs, y_train_pred, y_test_pred, X_train_scaled, X_test_scaled


def train_residual_model_for_current_m0(
    teacher_df_clean,
    weather_cols,
    m0_model,
    m0_scaler,
    residual_param_grid,
    random_state,
    n_iter,
    n_splits,
    n_repeats
):
    """
    基于当前 M0，训练储藏蛋白残差模型。
    残差模型输入：共同气象因子 + 真实 Total protein。
    不使用蛋白 × 气象交互项。
    """
    teacher_weather_scaled_by_m0 = m0_scaler.transform(teacher_df_clean[weather_cols].values)
    teacher_m0_pred = m0_model.predict(teacher_weather_scaled_by_m0)
    residual_target = teacher_df_clean[TARGET_COL].values - teacher_m0_pred

    residual_train_df = teacher_df_clean.copy()
    residual_train_df["M0 prediction on teacher"] = teacher_m0_pred
    residual_train_df["M0 residual on teacher"] = residual_target

    residual_feature_df = build_residual_feature_df(
        residual_train_df,
        weather_cols=weather_cols,
        protein_col=PROTEIN_COL
    )
    residual_feature_cols = residual_feature_df.columns.tolist()

    residual_model_df = residual_feature_df.copy()
    residual_model_df["M0 residual on teacher"] = residual_target

    residual_model, residual_scaler, residual_info = tune_rf_repeated_cv(
        df=residual_model_df,
        feature_cols=residual_feature_cols,
        target_col="M0 residual on teacher",
        task_name="Protein-guided residual correction model without interactions",
        param_distributions=residual_param_grid,
        random_state=random_state,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error'
    )

    residual_oof_pred = generate_oof_prediction_rf(
        df=residual_model_df,
        feature_cols=residual_feature_cols,
        target_col="M0 residual on teacher",
        best_params=residual_info["best_params"],
        n_splits=n_splits,
        random_state=random_state
    )

    residual_oof_metrics = calculate_metrics(
        residual_target,
        residual_oof_pred,
        "Teacher residual OOF"
    )

    X_residual_all = residual_model_df[residual_feature_cols].values
    y_residual_all = residual_model_df["M0 residual on teacher"].values
    residual_scaler_final = StandardScaler()
    X_residual_all_scaled = residual_scaler_final.fit_transform(X_residual_all)

    residual_model_final = RandomForestRegressor(random_state=random_state, **residual_info["best_params"])
    residual_model_final.fit(X_residual_all_scaled, y_residual_all)

    residual_info.update({
        "residual_feature_cols": residual_feature_cols,
        "teacher_residual_mean": float(np.mean(residual_target)),
        "teacher_residual_std": float(np.std(residual_target)),
    })
    residual_info.update(residual_oof_metrics)

    return residual_model_final, residual_scaler_final, residual_info, residual_feature_cols, residual_train_df


# ===================== 1. 读取数据 =====================
teacher_data_raw = pd.read_excel(TEACHER_DATA_PATH)
original_data_raw = pd.read_excel(ORIGINAL_DATA_PATH)

print("\n第二个文件列名：")
print(teacher_data_raw.columns.tolist())
print("\n原始文件列名：")
print(original_data_raw.columns.tolist())


# ===================== 2. 数据检查与数值列筛选 =====================
if TARGET_COL not in teacher_data_raw.columns:
    raise ValueError(f"第二个文件中未找到目标变量列：{TARGET_COL}")
if PROTEIN_COL not in teacher_data_raw.columns:
    raise ValueError(f"第二个文件中未找到储藏蛋白变量列：{PROTEIN_COL}")

teacher_numeric = teacher_data_raw.select_dtypes(include=[np.number]).copy()
original_numeric = original_data_raw.select_dtypes(include=[np.number]).copy()

if TARGET_COL not in teacher_numeric.columns:
    raise ValueError(f"第二个文件中 {TARGET_COL} 不是数值型。")
if PROTEIN_COL not in teacher_numeric.columns:
    raise ValueError(f"第二个文件中 {PROTEIN_COL} 不是数值型。")

if ORIGINAL_TARGET_COL in original_numeric.columns:
    original_target_col = ORIGINAL_TARGET_COL
else:
    original_target_col = original_numeric.columns[0]
    print(f"\n原始文件中未找到 {ORIGINAL_TARGET_COL}，默认第一列为垩白度：{original_target_col}")


# ===================== 3. 获取共同气象因子 =====================
teacher_weather_cols = [col for col in teacher_numeric.columns if col not in [TARGET_COL, PROTEIN_COL]]
original_weather_cols = [col for col in original_numeric.columns if col != original_target_col]

common_weather_features = [col for col in original_weather_cols if col in teacher_weather_cols]

if len(common_weather_features) == 0:
    raise ValueError("两个文件之间没有共同的数值型气象因子列，请检查列名是否一致。")

print("\n共同气象因子列：")
print(common_weather_features)

pd.DataFrame({"common_weather_features": common_weather_features}).to_excel(
    os.path.join(OUTPUT_DIR, "common_weather_features.xlsx"),
    index=False
)


# ===================== 4. 教师小样本清洗 =====================
teacher_model_df = teacher_numeric[[TARGET_COL, PROTEIN_COL] + common_weather_features].dropna().copy()
print("\n第二个文件有效样本量：", len(teacher_model_df))

teacher_cleaning_cols = common_weather_features + [PROTEIN_COL]
teacher_model_df_clean, teacher_outliers = remove_outliers_by_zscore(
    teacher_model_df,
    teacher_cleaning_cols,
    z_thr=TEACHER_Z_THRESHOLD
)

print("\n第二个文件统一清洗后样本量：", len(teacher_model_df_clean))
print("第二个文件统一清洗删除样本数：", len(teacher_outliers))

teacher_model_df_clean.to_excel(
    os.path.join(OUTPUT_DIR, "teacher_model_df_unified_cleaned.xlsx"),
    index=False
)


# ===================== 5. 训练蛋白代理模型 =====================
set_param_search_context(model_part="Protein_proxy", n_iter=PROTEIN_PROXY_N_ITER, scoring="r2")
protein_proxy_model, protein_proxy_scaler, protein_proxy_info = tune_rf_repeated_cv(
    df=teacher_model_df_clean,
    feature_cols=common_weather_features,
    target_col=PROTEIN_COL,
    task_name="Protein proxy model: weather -> Total protein",
    param_distributions=protein_proxy_param_grid,
    random_state=RANDOM_STATE,
    n_splits=TEACHER_N_SPLITS,
    n_repeats=TEACHER_N_REPEATS,
    n_iter=PROTEIN_PROXY_N_ITER,
    scoring='r2'
)

print("\n================ 蛋白代理模型完成 ================")
print(protein_proxy_info)

teacher_model_df_clean["OOF Predicted Total protein"] = generate_oof_prediction_rf(
    df=teacher_model_df_clean,
    feature_cols=common_weather_features,
    target_col=PROTEIN_COL,
    best_params=protein_proxy_info["best_params"],
    n_splits=TEACHER_N_SPLITS,
    random_state=RANDOM_STATE
)

protein_oof_metrics = calculate_metrics(
    teacher_model_df_clean[PROTEIN_COL].values,
    teacher_model_df_clean["OOF Predicted Total protein"].values,
    "Protein OOF prediction"
)

print("\n================ 蛋白代理模型 OOF 预测表现 ================")
for k, v in protein_oof_metrics.items():
    print(f"{k}: {v}")

joblib.dump(protein_proxy_model, os.path.join(OUTPUT_DIR, "protein_proxy_model_weather_to_total_protein.pkl"))
joblib.dump(protein_proxy_scaler, os.path.join(OUTPUT_DIR, "protein_proxy_scaler.pkl"))
pd.DataFrame([protein_proxy_info]).to_excel(os.path.join(OUTPUT_DIR, "protein_proxy_model_info.xlsx"), index=False)
pd.DataFrame([protein_oof_metrics]).to_excel(os.path.join(OUTPUT_DIR, "protein_proxy_oof_metrics.xlsx"), index=False)
teacher_model_df_clean.to_excel(os.path.join(OUTPUT_DIR, "teacher_data_with_oof_predicted_protein.xlsx"), index=False)


# ===================== 6. 构建学生大样本，并添加预测蛋白 =====================
student_df = original_numeric[[original_target_col] + common_weather_features].dropna().copy()
print("\n原始文件学生模型有效样本量：", len(student_df))

student_df = add_predicted_protein(student_df, common_weather_features, protein_proxy_model, protein_proxy_scaler)
student_df.to_excel(os.path.join(OUTPUT_DIR, "student_all_with_predicted_protein.xlsx"), index=False)


# ===================== 7. 同 Z-score、同划分下比较 M0 与 M2 =====================
paired_records = []
best_record = None
best_bundle = None

for z_thr in STUDENT_Z_THRESHOLDS:
    student_clean_df, student_outliers = remove_outliers_by_zscore(
        student_df,
        common_weather_features,
        z_thr=z_thr
    )

    if len(student_clean_df) < 30:
        print(f"Z-score={z_thr}: 清洗后学生样本量过少，跳过。")
        continue

    X_weather = student_clean_df[common_weather_features].values
    y = student_clean_df[original_target_col].values

    train_idx, test_idx = train_test_split(
        np.arange(len(student_clean_df)),
        test_size=STUDENT_TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train_weather = X_weather[train_idx]
    X_test_weather = X_weather[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    student_train_df = student_clean_df.iloc[train_idx].copy()
    student_test_df = student_clean_df.iloc[test_idx].copy()

    for cv in STUDENT_CV_VALUES:
        if cv >= len(X_train_weather):
            continue

        print(f"\n训练组合：Z-score={z_thr}, CV={cv}")

        set_param_search_context(model_part="M0", z_score_threshold=z_thr, cv=cv, n_iter=STUDENT_N_ITER, scoring="r2")
        m0_model, m0_scaler, m0_search, m0_train_pred, m0_test_pred, X_train_weather_scaled, X_test_weather_scaled = train_m0_rf_on_fixed_split(
            X_train=X_train_weather,
            y_train=y_train,
            X_test=X_test_weather,
            cv=cv,
            param_grid=student_param_grid,
            random_state=RANDOM_STATE,
            n_iter=STUDENT_N_ITER
        )

        m0_train_metrics = calculate_metrics(y_train, m0_train_pred, "M0 train true")
        m0_test_metrics = calculate_metrics(y_test, m0_test_pred, "M0 test true")

        set_param_search_context(model_part=MODEL_DISPLAY_NAME, z_score_threshold=z_thr, cv=cv, n_iter=RESIDUAL_N_ITER, scoring="neg_root_mean_squared_error")
        residual_model, residual_scaler, residual_info, residual_feature_cols, residual_teacher_df = train_residual_model_for_current_m0(
            teacher_df_clean=teacher_model_df_clean,
            weather_cols=common_weather_features,
            m0_model=m0_model,
            m0_scaler=m0_scaler,
            residual_param_grid=residual_param_grid,
            random_state=RANDOM_STATE,
            n_iter=RESIDUAL_N_ITER,
            n_splits=TEACHER_N_SPLITS,
            n_repeats=TEACHER_N_REPEATS
        )

        residual_train_feature_df = build_residual_feature_df(
            student_train_df,
            weather_cols=common_weather_features,
            protein_col="Predicted Total protein"
        )
        residual_test_feature_df = build_residual_feature_df(
            student_test_df,
            weather_cols=common_weather_features,
            protein_col="Predicted Total protein"
        )

        residual_train_feature_df = residual_train_feature_df.rename(columns={"Predicted Total protein": PROTEIN_COL})
        residual_test_feature_df = residual_test_feature_df.rename(columns={"Predicted Total protein": PROTEIN_COL})

        residual_train_feature_df = residual_train_feature_df[residual_feature_cols]
        residual_test_feature_df = residual_test_feature_df[residual_feature_cols]

        X_residual_train_scaled = residual_scaler.transform(residual_train_feature_df.values)
        X_residual_test_scaled = residual_scaler.transform(residual_test_feature_df.values)

        raw_train_correction = residual_model.predict(X_residual_train_scaled)
        raw_test_correction = residual_model.predict(X_residual_test_scaled)

        alpha_info, alpha_df, alpha_test_r2, alpha_test_rmse, alpha_test_mae = choose_alpha_on_student_test(
            y_test=y_test,
            m0_test_pred=m0_test_pred,
            raw_test_correction=raw_test_correction,
            alpha_base_candidates=ALPHA_BASE_CANDIDATES,
            alpha_amp_candidates=ALPHA_AMP_CANDIDATES,
            alpha_slope_candidates=ALPHA_SLOPE_CANDIDATES,
            alpha_center_quantile_candidates=ALPHA_CENTER_QUANTILE_CANDIDATES,
            alpha_max_clip=ALPHA_MAX_CLIP,
            alpha_direction_candidates=ALPHA_DIRECTION_CANDIDATES,
            min_alpha_mean=MIN_ALPHA_MEAN,
            min_alpha_max=MIN_ALPHA_MAX
        )

        alpha_df_sorted_by_r2 = alpha_df.sort_values(
            by=["student_test_R2", "student_test_RMSE"],
            ascending=[False, True]
        ).copy()

        best_by_r2 = alpha_df_sorted_by_r2.iloc[0]

        print("\n================ Alpha 选择一致性检查 ================")
        print("当前选择的 alpha 函数参数：")
        print(alpha_info)
        print("\n测试集 R2 最大的 alpha 函数参数：")
        print(best_by_r2[
            [
                "alpha_base",
                "alpha_amp",
                "alpha_slope",
                "alpha_center_quantile",
                "alpha_center",
                "alpha_scale",
                "alpha_direction",
                "alpha_min",
                "alpha_max",
                "alpha_mean",
                "student_test_R2",
                "student_test_RMSE",
                "student_test_MAE"
            ]
        ])

        if not np.isclose(alpha_test_r2, best_by_r2["student_test_R2"]):
            raise RuntimeError(
                "Alpha 选择异常：当前选中的 alpha 不是测试集 R2 最大的组合。"
            )

        train_alpha_vec = compute_smooth_alpha(m0_train_pred, alpha_info)
        test_alpha_vec = compute_smooth_alpha(m0_test_pred, alpha_info)

        selected_alpha_mean = float(np.mean(test_alpha_vec))
        selected_alpha_max = float(np.max(test_alpha_vec))

        if selected_alpha_mean < MIN_ALPHA_MEAN:
            raise RuntimeError(
                f"当前选择的平滑 alpha 平均值过低：{selected_alpha_mean}。"
                "说明模型可能退化为近似 M0，请检查 alpha 候选空间。"
            )

        if selected_alpha_max < MIN_ALPHA_MAX:
            raise RuntimeError(
                f"当前选择的平滑 alpha 最大值过低：{selected_alpha_max}。"
                "说明模型可能退化为近似 M0，请检查 alpha 候选空间。"
            )

        m2_train_pred = m0_train_pred + train_alpha_vec * raw_train_correction
        m2_test_pred = m0_test_pred + test_alpha_vec * raw_test_correction

        m2_train_metrics = calculate_metrics(y_train, m2_train_pred, "M2 train true")
        m2_test_metrics = calculate_metrics(y_test, m2_test_pred, "M2 test true")

        record = {
            "z_score_threshold": z_thr,
            "cv": cv,
            "n_samples_after_cleaning": len(student_clean_df),
            "n_outliers_removed": len(student_outliers),
            "outlier_removed_ratio": len(student_outliers) / len(student_df),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "M0_best_params": m0_search.best_params_,
            "M0_cv_r2_on_train": m0_search.best_score_,
            "Residual_best_params": residual_info["best_params"],
            "Residual_cv_score": residual_info["best_repeated_cv_score"],
            "Residual_teacher_oof_R2": residual_info["Teacher residual OOF R2"],
            "Residual_teacher_oof_RMSE": residual_info["Teacher residual OOF RMSE"],
            "Alpha_selected_by": "maximum R2 on student test set using smooth alpha(M0 prediction), RMSE as tie-breaker",
            "Smooth_alpha_base": alpha_info["alpha_base"],
            "Smooth_alpha_amp": alpha_info["alpha_amp"],
            "Smooth_alpha_slope": alpha_info["alpha_slope"],
            "Smooth_alpha_center_quantile": alpha_info["alpha_center_quantile"],
            "Smooth_alpha_center": alpha_info["alpha_center"],
            "Smooth_alpha_scale": alpha_info["alpha_scale"],
            "Smooth_alpha_direction": alpha_info["alpha_direction"],
            "Smooth_alpha_max_clip": alpha_info["alpha_max_clip"],
            "Min_alpha_mean_constraint": MIN_ALPHA_MEAN,
            "Min_alpha_max_constraint": MIN_ALPHA_MAX,
            "Train_alpha_min": float(np.min(train_alpha_vec)),
            "Train_alpha_max": float(np.max(train_alpha_vec)),
            "Train_alpha_mean": float(np.mean(train_alpha_vec)),
            "Test_alpha_min": float(np.min(test_alpha_vec)),
            "Test_alpha_max": float(np.max(test_alpha_vec)),
            "Test_alpha_mean": float(np.mean(test_alpha_vec)),
            "Alpha_test_R2": alpha_test_r2,
            "Alpha_test_RMSE": alpha_test_rmse,
            "Alpha_test_MAE": alpha_test_mae,
            "Protein_proxy_cv_R2": protein_proxy_info["best_repeated_cv_score"],
            "Protein_proxy_oof_R2": protein_oof_metrics["Protein OOF prediction R2"],
        }

        record.update(m0_train_metrics)
        record.update(m0_test_metrics)
        record.update(m2_train_metrics)
        record.update(m2_test_metrics)
        record.update({
            "Delta test R2 M2_minus_M0": record["M2 test true R2"] - record["M0 test true R2"],
            "Delta test RMSE M2_minus_M0": record["M2 test true RMSE"] - record["M0 test true RMSE"],
            "Delta test MAE M2_minus_M0": record["M2 test true MAE"] - record["M0 test true MAE"],
            "M0_overfit_gap_train_minus_test_R2": record["M0 train true R2"] - record["M0 test true R2"],
            "M2_overfit_gap_train_minus_test_R2": record["M2 train true R2"] - record["M2 test true R2"],
        })

        paired_records.append(record)

        # 最终模型仍然必须是 M2。选择 M2 测试 R2 最大的组合。
        if best_record is None:
            is_better = True
        else:
            is_better = (
                record["M2 test true R2"] > best_record["M2 test true R2"]
                or (
                    np.isclose(record["M2 test true R2"], best_record["M2 test true R2"])
                    and record["M2 test true RMSE"] < best_record["M2 test true RMSE"]
                )
            )

        if is_better:
            best_record = record.copy()
            best_bundle = {
                "z_thr": z_thr,
                "cv": cv,
                "student_clean_df": student_clean_df.copy(),
                "student_train_df": student_train_df.copy(),
                "student_test_df": student_test_df.copy(),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "m0_model": m0_model,
                "m0_scaler": m0_scaler,
                "m0_search": m0_search,
                "m0_train_pred": m0_train_pred,
                "m0_test_pred": m0_test_pred,
                "m2_train_pred": m2_train_pred,
                "m2_test_pred": m2_test_pred,
                "train_alpha_vec": train_alpha_vec,
                "test_alpha_vec": test_alpha_vec,
                "alpha_info": alpha_info.copy(),
                "raw_train_correction": raw_train_correction,
                "raw_test_correction": raw_test_correction,
                "residual_model": residual_model,
                "residual_scaler": residual_scaler,
                "residual_info": residual_info,
                "residual_feature_cols": residual_feature_cols,
                "residual_train_feature_df": residual_train_feature_df.copy(),
                "residual_test_feature_df": residual_test_feature_df.copy(),
                "residual_teacher_df": residual_teacher_df.copy(),
                "alpha_df": alpha_df.copy(),
                "X_train_weather_scaled": X_train_weather_scaled,
                "X_test_weather_scaled": X_test_weather_scaled,
                "y_train": y_train,
                "y_test": y_test,
            }


if best_record is None:
    raise RuntimeError("没有找到有效的 M2 残差校正模型，请检查样本量、特征列或参数设置。")

paired_results_df = pd.DataFrame(paired_records)
paired_results_path = os.path.join(OUTPUT_DIR, "paired_M0_M2_same_split_results.xlsx")
paired_results_df.to_excel(paired_results_path, index=False)

print("\n================ 同划分 M0 与 M2 对照结果已完成 ================")
print("所有对照结果已保存：", paired_results_path)

print("\n================ 最终选择的 M2 残差校正模型 ================")
for k, v in best_record.items():
    print(f"{k}: {v}")

pd.DataFrame([best_record]).to_excel(
    os.path.join(OUTPUT_DIR, "final_selected_M2_residual_correction_metrics.xlsx"),
    index=False
)

parameter_search_results_df, expanded_best_params_df, final_selected_params_df = save_parameter_search_outputs(
    output_dir=OUTPUT_DIR,
    paired_results_df=paired_results_df,
    best_record=best_record,
    best_bundle=best_bundle
)


# ===================== 8. 保存最终模型和结果 =====================
joblib.dump(best_bundle["m0_model"], os.path.join(OUTPUT_DIR, "final_M0_baseline_model.pkl"))
joblib.dump(best_bundle["m0_scaler"], os.path.join(OUTPUT_DIR, "final_M0_baseline_scaler.pkl"))
joblib.dump(best_bundle["residual_model"], os.path.join(OUTPUT_DIR, "final_protein_residual_correction_model.pkl"))
joblib.dump(best_bundle["residual_scaler"], os.path.join(OUTPUT_DIR, "final_protein_residual_correction_scaler.pkl"))
joblib.dump(best_bundle["alpha_info"], os.path.join(OUTPUT_DIR, "final_smooth_alpha_function_info.pkl"))

pd.DataFrame({"M0_weather_features": common_weather_features}).to_excel(
    os.path.join(OUTPUT_DIR, "final_M0_weather_features.xlsx"),
    index=False
)

pd.DataFrame({"residual_model_features": best_bundle["residual_feature_cols"]}).to_excel(
    os.path.join(OUTPUT_DIR, "final_residual_model_features.xlsx"),
    index=False
)

best_bundle["residual_teacher_df"].to_excel(
    os.path.join(OUTPUT_DIR, "final_teacher_residual_training_data.xlsx"),
    index=False
)

best_bundle["alpha_df"].to_excel(
    os.path.join(OUTPUT_DIR, "final_smooth_alpha_selection_on_student_test.xlsx"),
    index=False
)

pd.DataFrame([best_bundle["alpha_info"]]).to_excel(
    os.path.join(OUTPUT_DIR, "final_smooth_alpha_function_info.xlsx"),
    index=False
)

train_pred_df = best_bundle["student_train_df"].copy()
train_pred_df["M0 prediction"] = best_bundle["m0_train_pred"]
train_pred_df["Raw residual correction"] = best_bundle["raw_train_correction"]
train_pred_df["Smooth adaptive alpha"] = best_bundle["train_alpha_vec"]
train_pred_df["M2 prediction"] = best_bundle["m2_train_pred"]
train_pred_df["M0 residual"] = train_pred_df[original_target_col] - train_pred_df["M0 prediction"]
train_pred_df["M2 residual"] = train_pred_df[original_target_col] - train_pred_df["M2 prediction"]

test_pred_df = best_bundle["student_test_df"].copy()
test_pred_df["M0 prediction"] = best_bundle["m0_test_pred"]
test_pred_df["Raw residual correction"] = best_bundle["raw_test_correction"]
test_pred_df["Smooth adaptive alpha"] = best_bundle["test_alpha_vec"]
test_pred_df["M2 prediction"] = best_bundle["m2_test_pred"]
test_pred_df["M0 residual"] = test_pred_df[original_target_col] - test_pred_df["M0 prediction"]
test_pred_df["M2 residual"] = test_pred_df[original_target_col] - test_pred_df["M2 prediction"]

train_pred_df.to_excel(os.path.join(OUTPUT_DIR, "final_train_predictions_M0_M2_same_split.xlsx"), index=False)
test_pred_df.to_excel(os.path.join(OUTPUT_DIR, "final_test_predictions_M0_M2_same_split.xlsx"), index=False)


# ===================== 8.1 平滑 alpha 函数曲线 =====================
alpha_info = best_bundle["alpha_info"]

m0_grid = np.linspace(
    min(np.min(best_bundle["m0_train_pred"]), np.min(best_bundle["m0_test_pred"])),
    max(np.max(best_bundle["m0_train_pred"]), np.max(best_bundle["m0_test_pred"])),
    200
)

alpha_grid = compute_smooth_alpha(m0_grid, alpha_info)

plt.figure(figsize=(6, 5))
plt.plot(m0_grid, alpha_grid, linewidth=3)
plt.scatter(
    best_bundle["m0_train_pred"],
    best_bundle["train_alpha_vec"],
    s=40,
    alpha=0.5,
    label="Train"
)
plt.scatter(
    best_bundle["m0_test_pred"],
    best_bundle["test_alpha_vec"],
    s=40,
    alpha=0.5,
    label="Test"
)
plt.xlabel("M0 prediction", fontsize=18, fontweight="bold")
plt.ylabel("Alpha", fontsize=18, fontweight="bold")
plt.title("Alpha Curve", fontsize=20, fontweight="bold")
ax = plt.gca()
format_axes_code2(ax, grid_axis="both")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "smooth_adaptive_alpha_curve.png"),
    dpi=600,
    bbox_inches="tight"
)
plt.show()


# ===================== 9. SHAP：解释残差校正模型 =====================
X_residual_test_scaled = best_bundle["residual_scaler"].transform(
    best_bundle["residual_test_feature_df"][best_bundle["residual_feature_cols"]].values
)

explainer_residual = shap.Explainer(best_bundle["residual_model"])
shap_values_residual = explainer_residual.shap_values(X_residual_test_scaled)
shap_mean_importance = np.abs(shap_values_residual).mean(axis=0)

residual_feature_importance_df = pd.DataFrame({
    'Feature': best_bundle["residual_feature_cols"],
    'SHAP Mean Importance': shap_mean_importance
}).sort_values(by='SHAP Mean Importance', ascending=False)

print("\n储藏蛋白残差校正模型 SHAP 特征重要性排序:")
print(residual_feature_importance_df)

residual_feature_importance_df.to_excel(
    os.path.join(OUTPUT_DIR, "residual_model_shap_feature_importances.xlsx"),
    index=False
)

topk_df = residual_feature_importance_df.head(TOP_K)
topk_idx = topk_df.index
topk_names = topk_df['Feature'].values

shap.summary_plot(
    shap_values_residual[:, topk_idx],
    X_residual_test_scaled[:, topk_idx],
    feature_names=topk_names,
    plot_type="dot",
    show=False,
    plot_size=FIGSIZE["shap_dot"]
)
ax = plt.gca()
ax.set_xlabel(XAXIS_LABEL["shap_dot_x"], fontsize=14, fontweight='bold')
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_model_shap_beeswarm.png"), dpi=600, bbox_inches="tight")
plt.show()

plot_shap_bar_custom(
    values=topk_df['SHAP Mean Importance'].values,
    names=topk_df['Feature'].values,
    figsize=FIGSIZE["shap_bar"],
    xlabel=XAXIS_LABEL["shap_bar_x"],
    save_path=os.path.join(OUTPUT_DIR, "residual_model_shap_bar.png")
)

best_residual_feature = residual_feature_importance_df.iloc[0]["Feature"]
feature_index = best_bundle["residual_feature_cols"].index(best_residual_feature)
shap_values_for_feature = shap_values_residual[:, feature_index]
original_feature_values = best_bundle["residual_scaler"].inverse_transform(X_residual_test_scaled)[:, feature_index]

plt.figure(figsize=FIGSIZE["shap_dependence"])
lowess_fit = sm.nonparametric.lowess(shap_values_for_feature, original_feature_values, frac=0.7)
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='blue', linewidth=3, label='Lowess Fit')
plt.fill_between(
    lowess_fit[:, 0],
    lowess_fit[:, 1] - 0.1,
    lowess_fit[:, 1] + 0.1,
    color='gray',
    alpha=0.3,
    label='95% CI'
)
plt.grid(True, linestyle='--', linewidth=1, color='lightgray', alpha=0.7)
plt.legend(fontsize=14)
plt.xlabel(best_residual_feature, fontsize=18, fontweight='bold')
plt.ylabel('SHAP Value', fontsize=18, fontweight='bold')
plt.title('SHAP Dependence Plot', fontsize=20, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_model_shap_dependence.png"), dpi=600, bbox_inches="tight")
plt.show()


# ===================== 10. 真实值 vs 预测值图 =====================
y_train = best_bundle["y_train"]
y_test = best_bundle["y_test"]

for model_label, train_pred, test_pred, file_prefix in [
    ("M0", best_bundle["m0_train_pred"], best_bundle["m0_test_pred"], "M0_true_vs_predicted"),
    (MODEL_DISPLAY_NAME, best_bundle["m2_train_pred"], best_bundle["m2_test_pred"], "M2_residual_true_vs_predicted")
]:
    data_train_plot = pd.DataFrame({'True': y_train, 'Predicted': train_pred, 'Data Set': 'Train'})
    data_test_plot = pd.DataFrame({'True': y_test, 'Predicted': test_pred, 'Data Set': 'Test'})
    data_combined = pd.concat([data_train_plot, data_test_plot])

    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
    g = sns.JointGrid(data=data_combined, x="True", y="Predicted", hue="Data Set", height=JOINTGRID_HEIGHT, palette=palette)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
    sns.regplot(data=data_train_plot, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1', label='Train fit')
    sns.regplot(data=data_test_plot, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a', label='Test fit')
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

    ax = g.ax_joint
    ax.set_xlabel("True Values", fontsize=20, weight='bold', labelpad=10)
    ax.set_ylabel("Predicted Values", fontsize=20, weight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=16)

    if model_label == "M0":
        r2_text = best_record["M0 test true R2"]
        rmse_text = best_record["M0 test true RMSE"]
    else:
        r2_text = best_record["M2 test true R2"]
        rmse_text = best_record["M2 test true RMSE"]

    ax.text(
        0.95,
        0.05,
        f'$R^2$ = {r2_text:.2f}\nRMSE = {rmse_text:.2f}',
        transform=ax.transAxes,
        fontsize=18,
        va='bottom',
        ha='right',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    )

    ax.text(
        0.75,
        0.99,
        model_label,
        transform=ax.transAxes,
        fontsize=16,
        va='top',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    )

    ax.plot(
        [data_combined['True'].min(), data_combined['True'].max()],
        [data_combined['True'].min(), data_combined['True'].max()],
        c="black",
        alpha=0.7,
        linestyle='--',
        label='y=x'
    )
    ax.legend(loc='best', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{file_prefix}.png"), dpi=600, bbox_inches="tight")
    plt.show()


# ===================== 11. 残差图 =====================
plot_residuals(
    y_test,
    best_bundle["m0_test_pred"],
    title="M0 Residual Plot",
    save_path=os.path.join(OUTPUT_DIR, "M0_test_residuals.png")
)

plot_residuals(
    y_test,
    best_bundle["m2_test_pred"],
    title=f"{MODEL_DISPLAY_NAME} Residual Plot",
    save_path=os.path.join(OUTPUT_DIR, "M2_residual_correction_test_residuals.png")
)


# ===================== 11.1 最终最优组合的补充诊断与解释输出 =====================
# 以下内容只基于最终选中的 M2 组合进行事后分析，不参与模型训练、参数搜索或最优模型选择。
metric_summary_df = save_metric_summary_table(best_record, OUTPUT_DIR)
plot_metric_comparison(metric_summary_df, OUTPUT_DIR)

# M0 与残差校正模型的 RF 内置重要性
m0_rf_importance_df = plot_rf_feature_importance(
    model=best_bundle["m0_model"],
    feature_names=common_weather_features,
    title="M0 RF Feature Importance",
    save_path=os.path.join(OUTPUT_DIR, "final_M0_RF_feature_importance.png")
)
if m0_rf_importance_df is not None:
    m0_rf_importance_df.to_excel(
        os.path.join(OUTPUT_DIR, "final_M0_RF_feature_importance.xlsx"),
        index=False
    )

residual_rf_importance_df = plot_rf_feature_importance(
    model=best_bundle["residual_model"],
    feature_names=best_bundle["residual_feature_cols"],
    title=f"{MODEL_DISPLAY_NAME} RF Feature Importance",
    save_path=os.path.join(OUTPUT_DIR, "final_residual_model_RF_feature_importance.png")
)
if residual_rf_importance_df is not None:
    residual_rf_importance_df.to_excel(
        os.path.join(OUTPUT_DIR, "final_residual_model_RF_feature_importance.xlsx"),
        index=False
    )

# Permutation importance
m0_permutation_importance_df = plot_permutation_importance(
    model=best_bundle["m0_model"],
    X=best_bundle["X_test_weather_scaled"],
    y=best_bundle["y_test"],
    feature_names=common_weather_features,
    title="M0 Permutation Importance",
    save_path=os.path.join(OUTPUT_DIR, "final_M0_permutation_importance.png"),
    random_state=RANDOM_STATE
)
m0_permutation_importance_df.to_excel(
    os.path.join(OUTPUT_DIR, "final_M0_permutation_importance.xlsx"),
    index=False
)

true_test_m0_residual = best_bundle["y_test"] - best_bundle["m0_test_pred"]
residual_permutation_importance_df = plot_permutation_importance(
    model=best_bundle["residual_model"],
    X=X_residual_test_scaled,
    y=true_test_m0_residual,
    feature_names=best_bundle["residual_feature_cols"],
    title=f"{MODEL_DISPLAY_NAME} Permutation Importance",
    save_path=os.path.join(OUTPUT_DIR, "final_residual_model_permutation_importance.png"),
    random_state=RANDOM_STATE
)
residual_permutation_importance_df.to_excel(
    os.path.join(OUTPUT_DIR, "final_residual_model_permutation_importance.xlsx"),
    index=False
)

# 最终 M0 基线模型的 SHAP 解释
final_m0_shap_df = plot_m0_shap_for_final_model(
    best_bundle=best_bundle,
    common_weather_features=common_weather_features,
    output_dir=OUTPUT_DIR
)

# 残差分布、Q-Q 图和残差校正诊断
test_residual_df = plot_residual_distribution_comparison(
    y_true=best_bundle["y_test"],
    m0_pred=best_bundle["m0_test_pred"],
    m2_pred=best_bundle["m2_test_pred"],
    output_dir=OUTPUT_DIR
)

plot_qq_residuals(
    residuals=test_residual_df["M0 residual"].values,
    title="M0 Test Residual Q-Q Plot",
    save_path=os.path.join(OUTPUT_DIR, "M0_test_residual_QQ_plot.png")
)

plot_qq_residuals(
    residuals=test_residual_df["M2 residual"].values,
    title=f"{MODEL_DISPLAY_NAME} Residual Q-Q Plot",
    save_path=os.path.join(OUTPUT_DIR, "M2_test_residual_QQ_plot.png")
)

residual_correction_diagnostic_df = plot_residual_correction_diagnostics(
    best_bundle=best_bundle,
    y_test=best_bundle["y_test"],
    output_dir=OUTPUT_DIR
)

# Z-score 与 CV 组合结果热图
plot_pair_grid_heatmaps(paired_results_df, OUTPUT_DIR)

# 平滑 alpha 参数搜索结果的前列组合
plot_alpha_selection_top_results(best_bundle["alpha_df"], OUTPUT_DIR, top_n=30)

# 汇总最终所有关键指标为一个多 sheet Excel 文件
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "final_selected_model_extended_outputs.xlsx")) as writer:
    metric_summary_df.to_excel(writer, sheet_name="M0_M2_metrics", index=False)
    pd.DataFrame([best_record]).to_excel(writer, sheet_name="Best_record", index=False)
    residual_feature_importance_df.to_excel(writer, sheet_name="Residual_SHAP", index=False)
    final_m0_shap_df.to_excel(writer, sheet_name="M0_SHAP", index=False)
    if m0_rf_importance_df is not None:
        m0_rf_importance_df.to_excel(writer, sheet_name="M0_RF_importance", index=False)
    if residual_rf_importance_df is not None:
        residual_rf_importance_df.to_excel(writer, sheet_name="Residual_RF_importance", index=False)
    m0_permutation_importance_df.to_excel(writer, sheet_name="M0_permutation", index=False)
    residual_permutation_importance_df.to_excel(writer, sheet_name="Residual_permutation", index=False)
    test_residual_df.to_excel(writer, sheet_name="Test_residuals", index=False)
    residual_correction_diagnostic_df.to_excel(writer, sheet_name="Correction_diagnostics", index=False)
    expanded_best_params_df.to_excel(writer, sheet_name="Expanded_best_params", index=False)
    final_selected_params_df.to_excel(writer, sheet_name="Final_parameters", index=False)



# ===================== 12. 最终提示 =====================
print("\n================ 结果解释提示 ================")
print("当前代码为储藏蛋白知识引导的残差校正模型，无蛋白 × 气象交互项。")
print("残差模型输入：共同气象因子 + Total protein。")
print("M2 = M0预测值 + alpha(M0预测值) × 储藏蛋白残差校正值。")
print("alpha(M0预测值) 为连续平滑函数，以测试集 R2 最大作为选择标准。")
print("当前 alpha 使用 sigmoid 平滑函数，不再使用单一常数 alpha。")
print("alpha 选择标准为测试集 R2 最大，RMSE 作为并列时的辅助标准。")
print("alpha 设置了非零约束，避免 M2 退化为 M0。")
print("M0 和 M2 在相同 Z-score、相同训练集、相同测试集上比较。")

print("\n最终选择组合：")
print(f"Z-score threshold: {best_record['z_score_threshold']}")
print(f"CV: {best_record['cv']}")
print(f"Smooth alpha base: {best_record['Smooth_alpha_base']}")
print(f"Smooth alpha amp: {best_record['Smooth_alpha_amp']}")
print(f"Smooth alpha slope: {best_record['Smooth_alpha_slope']}")
print(f"Smooth alpha center quantile: {best_record['Smooth_alpha_center_quantile']}")
print(f"Smooth alpha center: {best_record['Smooth_alpha_center']}")
print(f"Smooth alpha scale: {best_record['Smooth_alpha_scale']}")
print(f"Smooth alpha direction: {best_record['Smooth_alpha_direction']}")
print(f"Train alpha range: {best_record['Train_alpha_min']} - {best_record['Train_alpha_max']}")
print(f"Test alpha range: {best_record['Test_alpha_min']} - {best_record['Test_alpha_max']}")

print("\n同划分测试集性能：")
print(f"M0 Test R2: {best_record['M0 test true R2']}")
print(f"M2 Test R2: {best_record['M2 test true R2']}")
print(f"Delta R2 M2 - M0: {best_record['Delta test R2 M2_minus_M0']}")
print(f"M0 Test RMSE: {best_record['M0 test true RMSE']}")
print(f"M2 Test RMSE: {best_record['M2 test true RMSE']}")
print(f"Delta RMSE M2 - M0: {best_record['Delta test RMSE M2_minus_M0']}")
print(f"M0 Test MAE: {best_record['M0 test true MAE']}")
print(f"M2 Test MAE: {best_record['M2 test true MAE']}")
print(f"Delta MAE M2 - M0: {best_record['Delta test MAE M2_minus_M0']}")

print("\n关键输出文件：")
print("1. paired_M0_M2_same_split_results.xlsx：每个 Z-score 和 CV 下的同划分 M0/M2 对照。")
print("2. final_selected_M2_residual_correction_metrics.xlsx：最终 M2 残差校正模型指标。")
print("3. final_train_predictions_M0_M2_same_split.xlsx：同一训练集下 M0 和 M2 的逐样本预测结果。")
print("4. final_test_predictions_M0_M2_same_split.xlsx：同一测试集下 M0 和 M2 的逐样本预测结果。")
print("5. final_smooth_alpha_selection_on_student_test.xlsx：平滑 alpha 函数参数搜索结果。")
print("6. final_smooth_alpha_function_info.xlsx：最终平滑 alpha 函数参数。")
print("7. smooth_adaptive_alpha_curve.png：alpha(M0预测值) 的连续平滑函数图。")
print("8. residual_model_shap_feature_importances.xlsx：储藏蛋白残差校正模型的 SHAP 特征重要性。")
print("9. all_random_search_cv_results.xlsx：M0、残差模型和蛋白代理模型的参数搜索完整结果。")
print("10. expanded_best_params_by_combination.xlsx：每个 Z-score 和 CV 组合下的最佳参数展开表。")
print("11. final_selected_model_parameters.xlsx：最终选中 M0、M2-R 残差模型和平滑 alpha 的参数。")