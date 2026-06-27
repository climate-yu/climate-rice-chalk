# -*- coding: utf-8 -*-
"""
M3 独立样本预测、最佳模型检查与第一列回写完整代码

功能：
1. 读取已训练并保存的最佳 M3 模型组件；
2. 自动检查结果汇总表，辅助确认当前 best_M3 文件是否对应测试集 R2 最高的 M3；
3. 读取独立样本文件：
   D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\M3.xlsx
4. 独立样本第一列为空，第二列及之后为与原始建模样本一致的特征；
5. 根据这些特征预测第一列目标值；
6. 将预测值回写到第一列；
7. 额外输出预测诊断表，包括基础预测、预测蛋白、残差校正、alpha 和最终 M3 预测值。

前提：
必须已经成功运行过 M3 建模脚本，并在以下目录保存了模型文件：
D:\实验\毕业论文\第四章\3.模型汇总\模型汇总\08_saved_models

核心预测公式：
M3 prediction = M3 base prediction + alpha(M3 base prediction) × constrained residual correction
"""

import os
import json
import shutil
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# =========================================================
# 1. 路径配置
# =========================================================
INDEPENDENT_DATA_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\M3.xlsx"

MODEL_ROOT_DIR = r"D:\实验\毕业论文\第四章\3.模型汇总\模型汇总"
MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "08_saved_models")
TABLE_DIR = os.path.join(MODEL_ROOT_DIR, "01_tables")

FEATURE_FILE = os.path.join(TABLE_DIR, "M3_base_weather_features.xlsx")

# 是否直接覆盖原始 M3.xlsx
# True：预测值会写回原文件第一列，同时自动生成备份文件
# False：不覆盖原文件，只输出一个新文件
OVERWRITE_INPUT_FILE = True

OUTPUT_PREDICTED_FILE = INDEPENDENT_DATA_PATH.replace(".xlsx", "_M3预测回写结果.xlsx")
OUTPUT_DIAGNOSTIC_FILE = INDEPENDENT_DATA_PATH.replace(".xlsx", "_M3预测诊断结果.xlsx")
OUTPUT_METRIC_FILE = INDEPENDENT_DATA_PATH.replace(".xlsx", "_M3独立验证指标.xlsx")
OUTPUT_BEST_MODEL_CHECK_FILE = INDEPENDENT_DATA_PATH.replace(".xlsx", "_M3最佳模型检查结果.xlsx")
BACKUP_FILE = INDEPENDENT_DATA_PATH.replace(".xlsx", "_预测前备份.xlsx")


# =========================================================
# 2. 与原 M3 脚本一致的关键参数
# =========================================================
TARGET_COL = "Chalkiness degree"
PROTEIN_COL = "Total protein"

TMIN_COL = "TMIN"
TMIN_THRESHOLD = 20.0

TMIN_EXCESS_COL = "TMIN_excess_20"
TMIN_FLAG_COL = "TMIN_above_20_flag"
PREDICTED_PROTEIN_COL = "Predicted Total protein"
PROTEIN_RISK_COL = "Protein_Heat_Risk"


# =========================================================
# 3. 必须保留的模型类
#    用于正确读取 joblib 保存的自定义模型对象
# =========================================================
class ScaledRFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler
        self.model = model

    def fit(self, X, y, sample_weight=None):
        if self.scaler is None:
            self.scaler = StandardScaler()
        if self.model is None:
            self.model = RandomForestRegressor(random_state=42)

        X_scaled = self.scaler.fit_transform(X)

        if sample_weight is None:
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X_scaled, y, sample_weight=np.asarray(sample_weight, dtype=float))

        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y=None, sample_weight=None):
        return self

    def predict(self, X):
        return np.repeat(self.value, len(X))


class ConditionalResidualModel(BaseEstimator, RegressorMixin):
    def __init__(self, input_feature_cols, model_feature_cols, threshold_col, threshold_value, low_model, high_model):
        self.input_feature_cols = input_feature_cols
        self.model_feature_cols = model_feature_cols
        self.threshold_col = threshold_col
        self.threshold_value = threshold_value
        self.low_model = low_model
        self.high_model = high_model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.input_feature_cols)

        if self.threshold_col not in df.columns:
            raise ValueError(f"缺少残差模型阈值列：{self.threshold_col}")

        high_mask = df[self.threshold_col].values > self.threshold_value
        pred = np.zeros(len(df), dtype=float)

        if np.any(~high_mask):
            pred[~high_mask] = self.low_model.predict(
                df.loc[~high_mask, self.model_feature_cols].values
            )

        if np.any(high_mask):
            pred[high_mask] = self.high_model.predict(
                df.loc[high_mask, self.model_feature_cols].values
            )

        return pred


# =========================================================
# 4. 通用工具函数
# =========================================================
def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def check_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 不存在：{path}")


def add_tmin_threshold_features(df):
    df = df.copy()

    if TMIN_COL not in df.columns:
        raise ValueError(
            f"独立样本中缺少 {TMIN_COL} 列，无法构建 {TMIN_EXCESS_COL} 和 {TMIN_FLAG_COL}。"
        )

    df[TMIN_COL] = pd.to_numeric(df[TMIN_COL], errors="coerce")
    df[TMIN_EXCESS_COL] = np.maximum(df[TMIN_COL].values - TMIN_THRESHOLD, 0.0)
    df[TMIN_FLAG_COL] = (df[TMIN_COL].values > TMIN_THRESHOLD).astype(int)

    return df


def add_protein_heat_risk(df, protein_col):
    df = df.copy()

    if protein_col not in df.columns:
        raise ValueError(f"缺少蛋白输入列：{protein_col}")

    if TMIN_EXCESS_COL not in df.columns:
        raise ValueError(f"缺少阈值变量列：{TMIN_EXCESS_COL}")

    df[PROTEIN_RISK_COL] = df[protein_col].values * df[TMIN_EXCESS_COL].values

    return df


def stable_sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def compute_smooth_alpha(base_pred, alpha_info):
    """
    与原 M3 脚本保持一致的 alpha 计算方式。
    """
    base_pred = np.asarray(base_pred, dtype=float)

    scale = float(alpha_info.get("alpha_scale", 1.0))
    if scale <= 1e-12:
        scale = 1.0

    x = (base_pred - alpha_info["alpha_center"]) / scale

    if alpha_info.get("alpha_direction", "increasing") == "decreasing":
        x = -x

    smooth = stable_sigmoid(float(alpha_info["alpha_slope"]) * x)

    alpha_vec = float(alpha_info["alpha_base"]) + float(alpha_info["alpha_amp"]) * smooth

    alpha_max_clip = float(alpha_info.get("alpha_max_clip", 1.50))

    return np.clip(alpha_vec, 0.0, alpha_max_clip)


def compose_m3_prediction(base_pred, raw_correction, alpha_vec):
    """
    M3 = M3 base prediction + alpha × constrained residual correction
    """
    return (
        np.asarray(base_pred, dtype=float)
        + np.asarray(alpha_vec, dtype=float) * np.asarray(raw_correction, dtype=float)
    )


def force_numeric_features(df, feature_cols):
    """
    将模型输入特征强制转换为数值型。
    无法转换的内容会变成 NaN，后续不参与预测。
    """
    out = df.copy()

    for col in feature_cols:
        if col not in out.columns:
            raise ValueError(f"独立样本缺少模型需要的特征列：{col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def load_base_weather_features(independent_df):
    """
    优先读取原 M3 脚本输出的 M3_base_weather_features.xlsx。
    如果文件不存在，则默认使用独立样本第二列及之后的所有列作为基础特征。
    """
    if os.path.exists(FEATURE_FILE):
        feature_df = pd.read_excel(FEATURE_FILE)
        feature_df = clean_column_names(feature_df)

        if "M3_base_weather_features" in feature_df.columns:
            feature_cols = feature_df["M3_base_weather_features"].dropna().astype(str).str.strip().tolist()
        else:
            feature_cols = feature_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        print(f"已读取 M3 建模特征顺序文件：{FEATURE_FILE}")

    else:
        feature_cols = independent_df.columns[1:].astype(str).str.strip().tolist()
        print("未找到 M3_base_weather_features.xlsx，默认使用独立样本第二列及之后的列作为基础特征。")

    if len(feature_cols) == 0:
        raise ValueError("未能获得 M3 基础模型特征列。")

    return feature_cols


def build_residual_feature_df(pred_df, residual_feature_cols):
    """
    构建条件残差模型输入特征。
    独立样本中没有实测 Total protein 时，使用蛋白代理模型预测值作为残差模型的 Total protein 输入。
    """
    temp = pred_df.copy()

    temp[PROTEIN_COL] = temp[PREDICTED_PROTEIN_COL].values
    temp = add_protein_heat_risk(temp, PROTEIN_COL)

    residual_input = pd.DataFrame(index=temp.index)

    for col in residual_feature_cols:
        if col in temp.columns:
            residual_input[col] = temp[col].values
        else:
            raise ValueError(
                f"无法构建残差模型输入，缺少列：{col}。"
                f"请检查独立样本列名是否与原始建模数据一致。"
            )

    return residual_input


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# =========================================================
# 5. 最佳 M3 模型来源检查
# =========================================================
def check_best_m3_source(alpha_info):
    """
    检查 01_tables 中的结果表，尽量识别测试集 R2 最高的 M3 记录。
    该步骤不能直接读取 pkl 内部训练记录，但可辅助判断当前 best_M3 文件是否与汇总表中的最佳结果一致。
    """
    print("\n================ 当前 best_M3 模型来源检查 ================\n")

    possible_summary_files = [
        os.path.join(TABLE_DIR, "best_M3_summary.xlsx"),
        os.path.join(TABLE_DIR, "M3_best_summary.xlsx"),
        os.path.join(TABLE_DIR, "all_model_results.xlsx"),
        os.path.join(TABLE_DIR, "all_results.xlsx"),
        os.path.join(TABLE_DIR, "M3_all_results.xlsx"),
        os.path.join(TABLE_DIR, "model_results.xlsx"),
        os.path.join(TABLE_DIR, "all_M3_results.xlsx"),
        os.path.join(TABLE_DIR, "final_model_results.xlsx"),
    ]

    found_files = [p for p in possible_summary_files if os.path.exists(p)]

    check_records = []

    if len(found_files) == 0:
        print("未找到常见命名的 M3 结果汇总表。")
        print("这不影响预测，但无法在预测脚本中再次证明 pkl 是否对应所有历遍结果中的全局最佳 M3。")

        rec = {
            "check_status": "No summary table found",
            "summary_file": "",
            "identified_r2_column": "",
            "best_record_r2": np.nan,
            "note": "Prediction uses current best_M3 pkl files in 08_saved_models."
        }
        check_records.append(rec)

        pd.DataFrame(check_records).to_excel(OUTPUT_BEST_MODEL_CHECK_FILE, index=False)
        return pd.DataFrame(check_records)

    r2_candidates = [
        "M3 test R2",
        "M3 Test R2",
        "M3_test_R2",
        "M3 test_R2",
        "Test R2",
        "test_R2",
        "student_test_R2",
        "R2"
    ]

    rmse_candidates = [
        "M3 test RMSE",
        "M3 Test RMSE",
        "M3_test_RMSE",
        "Test RMSE",
        "test_RMSE",
        "student_test_RMSE",
        "RMSE"
    ]

    version_candidates = [
        "Version",
        "Model",
        "Model family",
        "model",
        "version"
    ]

    for summary_file in found_files:
        try:
            summary_df = pd.read_excel(summary_file)
            summary_df = clean_column_names(summary_df)

            r2_col = None
            for c in r2_candidates:
                if c in summary_df.columns:
                    r2_col = c
                    break

            rmse_col = None
            for c in rmse_candidates:
                if c in summary_df.columns:
                    rmse_col = c
                    break

            version_col = None
            for c in version_candidates:
                if c in summary_df.columns:
                    version_col = c
                    break

            if r2_col is None:
                check_records.append({
                    "check_status": "Summary found but R2 column not identified",
                    "summary_file": summary_file,
                    "identified_r2_column": "",
                    "best_record_r2": np.nan,
                    "identified_rmse_column": rmse_col if rmse_col else "",
                    "best_record_rmse": np.nan,
                    "version_column": version_col if version_col else "",
                    "best_record_version": "",
                    "note": "Please manually check this summary table."
                })
                continue

            temp_df = summary_df.copy()
            temp_df[r2_col] = pd.to_numeric(temp_df[r2_col], errors="coerce")
            temp_df = temp_df.dropna(subset=[r2_col])

            if len(temp_df) == 0:
                check_records.append({
                    "check_status": "R2 column identified but all values are NaN",
                    "summary_file": summary_file,
                    "identified_r2_column": r2_col,
                    "best_record_r2": np.nan,
                    "identified_rmse_column": rmse_col if rmse_col else "",
                    "best_record_rmse": np.nan,
                    "version_column": version_col if version_col else "",
                    "best_record_version": "",
                    "note": "No valid R2 values in this table."
                })
                continue

            best_idx = temp_df[r2_col].idxmax()
            best_row = temp_df.loc[best_idx]

            best_rmse = np.nan
            if rmse_col is not None and rmse_col in temp_df.columns:
                best_rmse = safe_float(best_row[rmse_col])

            best_version = ""
            if version_col is not None and version_col in temp_df.columns:
                best_version = str(best_row[version_col])

            rec = {
                "check_status": "Best record identified by highest R2",
                "summary_file": summary_file,
                "identified_r2_column": r2_col,
                "best_record_r2": safe_float(best_row[r2_col]),
                "identified_rmse_column": rmse_col if rmse_col else "",
                "best_record_rmse": best_rmse,
                "version_column": version_col if version_col else "",
                "best_record_version": best_version,
                "saved_alpha_base": safe_float(alpha_info.get("alpha_base", np.nan)),
                "saved_alpha_amp": safe_float(alpha_info.get("alpha_amp", np.nan)),
                "saved_alpha_slope": safe_float(alpha_info.get("alpha_slope", np.nan)),
                "saved_alpha_center": safe_float(alpha_info.get("alpha_center", np.nan)),
                "saved_alpha_center_quantile": safe_float(alpha_info.get("alpha_center_quantile", np.nan)),
                "saved_alpha_scale": safe_float(alpha_info.get("alpha_scale", np.nan)),
                "saved_alpha_direction": str(alpha_info.get("alpha_direction", "")),
                "note": "If the training script saved best_M3 from this best row, the independent prediction is the best M3 prediction."
            }

            check_records.append(rec)

            print(f"发现结果汇总表：{summary_file}")
            print(f"识别到 R2 列：{r2_col}")
            print(f"最高 R2：{rec['best_record_r2']}")
            if rmse_col is not None:
                print(f"对应 RMSE：{rec['best_record_rmse']}")
            if best_version != "":
                print(f"对应版本：{best_version}")

        except Exception as e:
            check_records.append({
                "check_status": "Failed to read summary table",
                "summary_file": summary_file,
                "identified_r2_column": "",
                "best_record_r2": np.nan,
                "identified_rmse_column": "",
                "best_record_rmse": np.nan,
                "version_column": "",
                "best_record_version": "",
                "note": str(e)
            })

    check_df = pd.DataFrame(check_records)
    check_df.to_excel(OUTPUT_BEST_MODEL_CHECK_FILE, index=False)

    print(f"\n最佳模型检查结果已输出：{OUTPUT_BEST_MODEL_CHECK_FILE}")
    print("\n说明：预测代码读取的是当前 08_saved_models 中的 best_M3 文件。")
    print("如果这些文件由训练脚本按最高测试集 R2 保存，则本次结果就是最佳 M3 的独立预测结果。")
    print("\n========================================================\n")

    return check_df


# =========================================================
# 6. 主程序
# =========================================================
def main():
    print("\n================ M3 独立样本预测开始 ================\n")

    # -----------------------------
    # 6.1 检查输入文件和模型文件
    # -----------------------------
    check_file_exists(INDEPENDENT_DATA_PATH, "独立样本文件")

    model_files = {
        "protein_model": os.path.join(MODEL_DIR, "protein_proxy_model.pkl"),
        "protein_scaler": os.path.join(MODEL_DIR, "protein_proxy_scaler.pkl"),
        "m3_base_model": os.path.join(MODEL_DIR, "best_M3_base_model.pkl"),
        "m3_base_scaler": os.path.join(MODEL_DIR, "best_M3_base_scaler.pkl"),
        "residual_model": os.path.join(MODEL_DIR, "best_M3_conditional_residual_model.pkl"),
        "alpha_info": os.path.join(MODEL_DIR, "best_M3_alpha_info.pkl"),
    }

    for name, path in model_files.items():
        check_file_exists(path, name)

    print("模型文件检查完成。")
    for name, path in model_files.items():
        print(f"{name}: {path}")

    # -----------------------------
    # 6.2 读取模型
    # -----------------------------
    protein_model = joblib.load(model_files["protein_model"])
    protein_scaler = joblib.load(model_files["protein_scaler"])

    m3_base_model = joblib.load(model_files["m3_base_model"])
    m3_base_scaler = joblib.load(model_files["m3_base_scaler"])

    residual_model = joblib.load(model_files["residual_model"])
    alpha_info = joblib.load(model_files["alpha_info"])

    print("\n模型读取完成。")

    print("\n当前读取的 alpha 参数：")
    print(json.dumps(alpha_info, ensure_ascii=False, indent=2))

    # -----------------------------
    # 6.3 检查当前 best_M3 的来源
    # -----------------------------
    check_best_m3_source(alpha_info)

    # -----------------------------
    # 6.4 读取独立样本
    # -----------------------------
    raw_df = pd.read_excel(INDEPENDENT_DATA_PATH)
    raw_df = clean_column_names(raw_df)

    if raw_df.shape[1] < 2:
        raise ValueError("独立样本表至少应包含第一列目标列和第二列及之后的特征列。")

    target_col = raw_df.columns[0]

    print("\n独立样本读取完成。")
    print(f"独立样本行数：{len(raw_df)}")
    print(f"第一列将被回写为预测值：{target_col}")

    # -----------------------------
    # 6.5 读取基础气象特征顺序
    # -----------------------------
    base_weather_features = load_base_weather_features(raw_df)

    print("\nM3 基础模型使用的特征列如下：")
    for i, col in enumerate(base_weather_features, start=1):
        print(f"{i}. {col}")

    # -----------------------------
    # 6.6 准备预测数据
    # -----------------------------
    pred_df = raw_df.copy()
    pred_df = force_numeric_features(pred_df, base_weather_features)

    if TMIN_COL not in pred_df.columns:
        raise ValueError(f"独立样本中缺少 {TMIN_COL}。")

    pred_df = add_tmin_threshold_features(pred_df)

    # 只对基础特征完整的样本进行预测
    valid_mask = pred_df[base_weather_features].notna().all(axis=1)

    n_valid = int(valid_mask.sum())
    n_invalid = int((~valid_mask).sum())

    print(f"\n可预测样本数：{n_valid}")
    print(f"因基础特征缺失无法预测样本数：{n_invalid}")

    if n_valid == 0:
        raise ValueError("没有可预测样本。请检查独立样本特征列名、数值格式和缺失值。")

    valid_df = pred_df.loc[valid_mask].copy()

    # -----------------------------
    # 6.7 M3 基础模型预测
    # -----------------------------
    X_weather = valid_df[base_weather_features].values

    m3_base_pred = m3_base_model.predict(
        m3_base_scaler.transform(X_weather)
    )

    # -----------------------------
    # 6.8 蛋白代理模型预测
    # -----------------------------
    predicted_protein = protein_model.predict(
        protein_scaler.transform(X_weather)
    )

    valid_df[PREDICTED_PROTEIN_COL] = predicted_protein

    # -----------------------------
    # 6.9 条件残差模型预测
    # -----------------------------
    if not hasattr(residual_model, "input_feature_cols"):
        raise AttributeError(
            "残差模型中没有 input_feature_cols 属性，无法自动构建残差模型输入特征。"
        )

    residual_feature_cols = list(residual_model.input_feature_cols)

    print("\n条件残差模型使用的输入特征列如下：")
    for i, col in enumerate(residual_feature_cols, start=1):
        print(f"{i}. {col}")

    residual_feature_df = build_residual_feature_df(
        pred_df=valid_df,
        residual_feature_cols=residual_feature_cols
    )

    raw_residual_correction = residual_model.predict(residual_feature_df.values)

    # -----------------------------
    # 6.10 alpha 计算和 M3 最终预测
    # -----------------------------
    adaptive_alpha = compute_smooth_alpha(m3_base_pred, alpha_info)

    m3_prediction = compose_m3_prediction(
        base_pred=m3_base_pred,
        raw_correction=raw_residual_correction,
        alpha_vec=adaptive_alpha
    )

    # -----------------------------
    # 6.11 回写预测结果
    # -----------------------------
    output_df = raw_df.copy()

    # 可预测样本写入预测值；不可预测样本保持原值
    output_df.loc[valid_mask, target_col] = m3_prediction

    # -----------------------------
    # 6.12 输出诊断结果
    # -----------------------------
    diagnostic_df = raw_df.copy()
    diagnostic_df["M3_valid_for_prediction"] = valid_mask.values
    diagnostic_df["M3_base_prediction"] = np.nan
    diagnostic_df[PREDICTED_PROTEIN_COL] = np.nan
    diagnostic_df["Raw residual correction"] = np.nan
    diagnostic_df["Adaptive alpha"] = np.nan
    diagnostic_df["M3 prediction"] = np.nan
    diagnostic_df[TMIN_EXCESS_COL] = np.nan
    diagnostic_df[TMIN_FLAG_COL] = np.nan
    diagnostic_df[PROTEIN_RISK_COL] = np.nan

    diagnostic_df.loc[valid_mask, "M3_base_prediction"] = m3_base_pred
    diagnostic_df.loc[valid_mask, PREDICTED_PROTEIN_COL] = predicted_protein
    diagnostic_df.loc[valid_mask, "Raw residual correction"] = raw_residual_correction
    diagnostic_df.loc[valid_mask, "Adaptive alpha"] = adaptive_alpha
    diagnostic_df.loc[valid_mask, "M3 prediction"] = m3_prediction
    diagnostic_df.loc[valid_mask, TMIN_EXCESS_COL] = valid_df[TMIN_EXCESS_COL].values
    diagnostic_df.loc[valid_mask, TMIN_FLAG_COL] = valid_df[TMIN_FLAG_COL].values
    diagnostic_df.loc[valid_mask, PROTEIN_RISK_COL] = (
        predicted_protein * valid_df[TMIN_EXCESS_COL].values
    )

    # 如果第一列有实测值，则保留实测值和残差
    y_true_numeric = pd.to_numeric(raw_df[target_col], errors="coerce")
    diagnostic_df["Observed value"] = y_true_numeric
    diagnostic_df["Prediction residual"] = y_true_numeric - diagnostic_df["M3 prediction"]
    diagnostic_df["Absolute prediction residual"] = np.abs(diagnostic_df["Prediction residual"])

    # -----------------------------
    # 6.13 保存预测结果
    # -----------------------------
    diagnostic_df.to_excel(OUTPUT_DIAGNOSTIC_FILE, index=False)
    output_df.to_excel(OUTPUT_PREDICTED_FILE, index=False)

    print(f"\n已输出预测回写结果文件：{OUTPUT_PREDICTED_FILE}")
    print(f"已输出预测诊断结果文件：{OUTPUT_DIAGNOSTIC_FILE}")

    if OVERWRITE_INPUT_FILE:
        if not os.path.exists(BACKUP_FILE):
            shutil.copy2(INDEPENDENT_DATA_PATH, BACKUP_FILE)
            print(f"已备份原始文件：{BACKUP_FILE}")
        else:
            print(f"备份文件已存在，未重复备份：{BACKUP_FILE}")

        output_df.to_excel(INDEPENDENT_DATA_PATH, index=False)
        print(f"已将预测结果回写到原始文件第一列：{INDEPENDENT_DATA_PATH}")
    else:
        print("当前设置为不覆盖原始文件，只输出新文件。")

    # -----------------------------
    # 6.14 如果第一列存在实测值，则计算独立验证指标
    # -----------------------------
    metric_mask = valid_mask.values & y_true_numeric.notna().values

    if metric_mask.sum() >= 2:
        y_true = y_true_numeric.loc[metric_mask].values.astype(float)
        y_pred = diagnostic_df.loc[metric_mask, "M3 prediction"].values.astype(float)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        metric_df = pd.DataFrame([{
            "Model": "M3 independent validation",
            "n": int(metric_mask.sum()),
            "R2": float(r2),
            "RMSE": float(rmse),
            "MSE": float(mse),
            "MAE": float(mae),
            "Independent_data_path": INDEPENDENT_DATA_PATH,
            "Model_dir": MODEL_DIR
        }])

        metric_df.to_excel(OUTPUT_METRIC_FILE, index=False)

        print("\n检测到第一列存在实测值，已计算独立验证指标：")
        print(metric_df)
        print(f"指标文件：{OUTPUT_METRIC_FILE}")

    else:
        print("\n第一列没有足够实测值，本次仅进行预测回写，不计算 R2、RMSE、MSE、MAE。")

    # -----------------------------
    # 6.15 输出简单统计
    # -----------------------------
    pred_summary = pd.DataFrame([{
        "n_total": int(len(raw_df)),
        "n_valid_prediction": int(n_valid),
        "n_invalid_prediction": int(n_invalid),
        "M3_prediction_min": float(np.nanmin(m3_prediction)),
        "M3_prediction_max": float(np.nanmax(m3_prediction)),
        "M3_prediction_mean": float(np.nanmean(m3_prediction)),
        "M3_prediction_std": float(np.nanstd(m3_prediction)),
        "base_prediction_mean": float(np.nanmean(m3_base_pred)),
        "raw_correction_mean": float(np.nanmean(raw_residual_correction)),
        "alpha_mean": float(np.nanmean(adaptive_alpha)),
        "alpha_min": float(np.nanmin(adaptive_alpha)),
        "alpha_max": float(np.nanmax(adaptive_alpha))
    }])

    summary_file = INDEPENDENT_DATA_PATH.replace(".xlsx", "_M3预测统计摘要.xlsx")
    pred_summary.to_excel(summary_file, index=False)

    print(f"\n预测统计摘要已输出：{summary_file}")
    print(pred_summary)

    print("\n================ M3 独立样本预测完成 ================\n")


if __name__ == "__main__":
    main()