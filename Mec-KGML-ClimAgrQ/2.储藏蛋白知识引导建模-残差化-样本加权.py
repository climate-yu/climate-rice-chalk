# -*- coding: utf-8 -*-
"""
M2 storage-protein residual correction model: sample weighting on residual correction, grid-search version.

Main purpose
1. For each Z-score × CV combination, first build WR0, the ordinary protein-guided residual correction model:
      M2 = M0 prediction + alpha(M0 prediction) × residual correction.
2. Then test sample weighting only on the residual model training process or on both residual model training and alpha selection.
3. All weighted versions are compared with WR0 under the same Z-score, the same CV, the same cleaned samples, and the same train/test split.
4. TMIN_excess_20, TMIN_above_20_flag, and Protein_Heat_Risk are used only for sample-weight construction.
   They are not used as M0, protein-proxy, or residual-model input features.
5. The script loops over:
      Z-score = [2, 3, 4]
      CV      = [4, 7, 10]
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# =========================================================
# 1. Configuration
# =========================================================
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\M2_残差校正基础上_样本加权检验结果_网格筛选_特征对齐版"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "Chalkiness degree"
PROTEIN_COL = "Total protein"
ORIGINAL_TARGET_COL = "Chalkiness degree"

TMIN_COL = "TMIN"
TMIN_THRESHOLD = 20.0
TMIN_EXCESS_COL = "TMIN_excess_20"
TMIN_FLAG_COL = "TMIN_above_20_flag"
PREDICTED_PROTEIN_COL = "Predicted Total protein"
PROTEIN_RISK_COL = "Protein_Heat_Risk"

# These variables are used only for sample-weight construction, not as model features.
WEIGHT_ONLY_COLS = [TMIN_EXCESS_COL, TMIN_FLAG_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL]

STUDENT_TEST_SIZE = 0.30
RANDOM_STATE = 42
TEACHER_Z_THRESHOLD = 4
TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10

STUDENT_Z_THRESHOLDS = [2, 3, 4]
STUDENT_CV_VALUES = [4, 7, 10]

PROTEIN_PROXY_N_ITER = 40
RESIDUAL_N_ITER = 40
STUDENT_N_ITER = 100

WEIGHT_SCHEMES = ["binary_tmin", "excess_tmin", "protein_heat_risk"]
WEIGHT_BETAS = [0.25, 0.5, 1.0, 2.0]
WEIGHT_LOCATIONS = ["residual_only", "residual_and_alpha"]

ALPHA_BASE_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
ALPHA_AMP_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
ALPHA_SLOPE_CANDIDATES = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
ALPHA_CENTER_QUANTILE_CANDIDATES = [0.35, 0.45, 0.50, 0.60, 0.70, 0.80]
ALPHA_DIRECTION_CANDIDATES = ["increasing"]
ALPHA_MAX_CLIP = 1.50
MIN_ALPHA_MEAN = 0.03
MIN_ALPHA_MAX = 0.05

student_param_grid = {
    "n_estimators": [2, 5, 10, 20, 30],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 7],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": ["log2", "sqrt"],
}

protein_proxy_param_grid = {
    "n_estimators": [20, 30, 50, 100, 200],
    "max_depth": [2, 3, 5, 7, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [2, 3, 4, 5],
    "max_features": ["sqrt", "log2"],
}

residual_param_grid = {
    "n_estimators": [20, 30, 50, 100],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [3, 4, 5, 8, 10],
    "max_features": ["sqrt", "log2"],
}

# =========================================================
# 2. Utility functions
# =========================================================
def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def add_tmin_threshold_features(df):
    df = df.copy()
    if TMIN_COL not in df.columns:
        raise ValueError(f"Missing {TMIN_COL}; cannot construct TMIN threshold variables.")
    df[TMIN_EXCESS_COL] = np.maximum(df[TMIN_COL].values - TMIN_THRESHOLD, 0.0)
    df[TMIN_FLAG_COL] = (df[TMIN_COL].values > TMIN_THRESHOLD).astype(int)
    return df


def add_protein_heat_risk(df, protein_col):
    df = df.copy()
    df[PROTEIN_RISK_COL] = df[protein_col].values * df[TMIN_EXCESS_COL].values
    return df


def normalize_0_1(x):
    x = np.asarray(x, dtype=float)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)


def build_sample_weight(df, scheme, beta, protein_col=None):
    if scheme == "none" or beta == 0:
        return np.ones(len(df), dtype=float)
    if scheme == "binary_tmin":
        signal = (df[TMIN_COL].values > TMIN_THRESHOLD).astype(float)
    elif scheme == "excess_tmin":
        signal = normalize_0_1(df[TMIN_EXCESS_COL].values)
    elif scheme == "protein_heat_risk":
        if protein_col is None or protein_col not in df.columns:
            raise ValueError("protein_heat_risk weighting requires a valid protein_col.")
        signal = normalize_0_1(df[protein_col].values * df[TMIN_EXCESS_COL].values)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")
    return 1.0 + beta * signal


def summarize_weight(w):
    w = np.asarray(w, dtype=float)
    return {
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_mean": float(np.mean(w)),
        "weight_std": float(np.std(w)),
    }


def remove_outliers_by_zscore(df, feature_cols, z_thr):
    X_df = df[feature_cols].copy()
    X_std = X_df.std(axis=0).replace(0, np.nan)
    z_scores = np.abs((X_df - X_df.mean(axis=0)) / X_std)
    z_scores = z_scores.fillna(0)
    outliers = np.where(np.any(z_scores >= z_thr, axis=1))[0]
    return df.drop(df.index[outliers]).copy(), outliers


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix} R2": float(r2_score(y_true, y_pred)),
        f"{prefix} RMSE": rmse_score(y_true, y_pred),
        f"{prefix} MSE": float(mean_squared_error(y_true, y_pred)),
        f"{prefix} MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def weighted_r2(y_true, y_pred, sample_weight):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    y_bar = np.sum(w * y_true) / np.sum(w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_bar) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def weighted_rmse(y_true, y_pred, sample_weight):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w)))


def weighted_mae(y_true, y_pred, sample_weight):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def stable_sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def compute_smooth_alpha(m0_pred, alpha_info):
    m0_pred = np.asarray(m0_pred, dtype=float)
    scale = float(alpha_info.get("alpha_scale", 1.0))
    if scale <= 1e-12:
        scale = 1.0
    x = (m0_pred - alpha_info["alpha_center"]) / scale
    if alpha_info["alpha_direction"] == "decreasing":
        x = -x
    smooth = stable_sigmoid(alpha_info["alpha_slope"] * x)
    alpha_vec = alpha_info["alpha_base"] + alpha_info["alpha_amp"] * smooth
    return np.clip(alpha_vec, 0.0, alpha_info["alpha_max_clip"])


def choose_alpha_on_student_test(y_test, m0_test_pred, raw_test_correction, alpha_selection_weight=None, selection_metric="r2"):
    records = []
    best_alpha_info = None
    best_score = -np.inf
    best_rmse = np.inf

    m0_test_pred = np.asarray(m0_test_pred, dtype=float)
    raw_test_correction = np.asarray(raw_test_correction, dtype=float)
    if alpha_selection_weight is None:
        alpha_selection_weight = np.ones(len(y_test), dtype=float)
    else:
        alpha_selection_weight = np.asarray(alpha_selection_weight, dtype=float)

    alpha_scale = float(np.std(m0_test_pred))
    if alpha_scale <= 1e-12:
        alpha_scale = 1.0
    center_dict = {q: float(np.quantile(m0_test_pred, q)) for q in ALPHA_CENTER_QUANTILE_CANDIDATES}

    for alpha_direction in ALPHA_DIRECTION_CANDIDATES:
        for alpha_base in ALPHA_BASE_CANDIDATES:
            for alpha_amp in ALPHA_AMP_CANDIDATES:
                for alpha_slope in ALPHA_SLOPE_CANDIDATES:
                    for center_q, alpha_center in center_dict.items():
                        alpha_info = {
                            "alpha_base": float(alpha_base),
                            "alpha_amp": float(alpha_amp),
                            "alpha_slope": float(alpha_slope),
                            "alpha_center_quantile": float(center_q),
                            "alpha_center": float(alpha_center),
                            "alpha_scale": float(alpha_scale),
                            "alpha_direction": alpha_direction,
                            "alpha_max_clip": float(ALPHA_MAX_CLIP),
                        }
                        alpha_vec = compute_smooth_alpha(m0_test_pred, alpha_info)
                        alpha_min = float(np.min(alpha_vec))
                        alpha_max = float(np.max(alpha_vec))
                        alpha_mean = float(np.mean(alpha_vec))
                        if alpha_mean < MIN_ALPHA_MEAN or alpha_max < MIN_ALPHA_MAX:
                            continue

                        pred = m0_test_pred + alpha_vec * raw_test_correction
                        normal_r2 = float(r2_score(y_test, pred))
                        normal_rmse = rmse_score(y_test, pred)
                        normal_mae = float(mean_absolute_error(y_test, pred))
                        wr2 = weighted_r2(y_test, pred, alpha_selection_weight)
                        wrmse = weighted_rmse(y_test, pred, alpha_selection_weight)
                        wmae = weighted_mae(y_test, pred, alpha_selection_weight)

                        if selection_metric == "weighted_r2":
                            score = wr2
                            tie_rmse = wrmse
                        else:
                            score = normal_r2
                            tie_rmse = normal_rmse

                        rec = {
                            **alpha_info,
                            "alpha_min": alpha_min,
                            "alpha_max": alpha_max,
                            "alpha_mean": alpha_mean,
                            "student_test_R2": normal_r2,
                            "student_test_RMSE": normal_rmse,
                            "student_test_MAE": normal_mae,
                            "student_test_weighted_R2": wr2,
                            "student_test_weighted_RMSE": wrmse,
                            "student_test_weighted_MAE": wmae,
                            "selection_score": score,
                        }
                        records.append(rec)

                        if (score > best_score) or (np.isclose(score, best_score) and tie_rmse < best_rmse):
                            best_score = score
                            best_rmse = tie_rmse
                            best_alpha_info = alpha_info.copy()

    if best_alpha_info is None:
        raise RuntimeError("No non-zero alpha function satisfies the constraints.")
    alpha_df = pd.DataFrame(records).sort_values(by=["selection_score", "student_test_RMSE"], ascending=[False, True])
    return best_alpha_info, alpha_df


def tune_rf_repeated_cv(df, feature_cols, target_col, param_distributions, n_splits, n_repeats, n_iter, scoring="r2", sample_weight=None):
    X = df[feature_cols].values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
    rs.fit(X_scaled, y, **fit_kwargs)
    info = {
        "best_repeated_cv_score": float(rs.best_score_),
        "best_params": rs.best_params_,
        "n_samples": len(df),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "scoring": scoring,
        "weighted_training": sample_weight is not None,
    }
    return rs.best_estimator_, scaler, info


def generate_oof_prediction_rf(df, feature_cols, target_col, best_params, n_splits=3, sample_weight=None):
    X = df[feature_cols].values
    y = df[target_col].values
    oof_pred = np.zeros(len(df), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, valid_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_valid = scaler.transform(X[valid_idx])
        model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
        if sample_weight is not None:
            model.fit(X_train, y[train_idx], sample_weight=np.asarray(sample_weight)[train_idx])
        else:
            model.fit(X_train, y[train_idx])
        oof_pred[valid_idx] = model.predict(X_valid)
    return oof_pred


def train_m0_rf_on_fixed_split(X_train, y_train, X_test, cv):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=student_param_grid,
        n_iter=STUDENT_N_ITER,
        cv=cv,
        scoring="r2",
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rs.fit(X_train_scaled, y_train)
    model = rs.best_estimator_
    return model, scaler, rs, model.predict(X_train_scaled), model.predict(X_test_scaled)


def build_residual_feature_df(df, weather_cols, protein_col):
    out = df[weather_cols].copy()
    out[PROTEIN_COL] = df[protein_col].values
    return out


def train_residual_model_for_current_m0(teacher_df_clean, weather_cols, m0_model, m0_scaler, residual_sample_weight=None, task_suffix=""):
    teacher_m0_pred = m0_model.predict(m0_scaler.transform(teacher_df_clean[weather_cols].values))
    residual_target = teacher_df_clean[TARGET_COL].values - teacher_m0_pred

    residual_train_df = teacher_df_clean.copy()
    residual_train_df["M0 prediction on teacher"] = teacher_m0_pred
    residual_train_df["M0 residual on teacher"] = residual_target

    residual_feature_df = build_residual_feature_df(residual_train_df, weather_cols, PROTEIN_COL)
    residual_feature_cols = residual_feature_df.columns.tolist()
    residual_model_df = residual_feature_df.copy()
    residual_model_df["M0 residual on teacher"] = residual_target

    residual_model, residual_scaler, residual_info = tune_rf_repeated_cv(
        residual_model_df,
        residual_feature_cols,
        "M0 residual on teacher",
        residual_param_grid,
        n_splits=TEACHER_N_SPLITS,
        n_repeats=TEACHER_N_REPEATS,
        n_iter=RESIDUAL_N_ITER,
        scoring="neg_root_mean_squared_error",
        sample_weight=residual_sample_weight,
    )
    residual_info["task_name"] = "Protein-guided residual correction model " + task_suffix

    residual_oof_pred = generate_oof_prediction_rf(
        residual_model_df,
        residual_feature_cols,
        "M0 residual on teacher",
        residual_info["best_params"],
        n_splits=TEACHER_N_SPLITS,
        sample_weight=residual_sample_weight,
    )
    residual_oof_metrics = calculate_metrics(residual_target, residual_oof_pred, "Teacher residual OOF")

    # Fit final residual model on all teacher samples.
    X_all = residual_model_df[residual_feature_cols].values
    y_all = residual_model_df["M0 residual on teacher"].values
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, **residual_info["best_params"])
    if residual_sample_weight is not None:
        final_model.fit(X_all_scaled, y_all, sample_weight=np.asarray(residual_sample_weight, dtype=float))
    else:
        final_model.fit(X_all_scaled, y_all)

    residual_info.update({
        "residual_feature_cols": residual_feature_cols,
        "teacher_residual_mean": float(np.mean(residual_target)),
        "teacher_residual_std": float(np.std(residual_target)),
    })
    residual_info.update(residual_oof_metrics)
    return final_model, final_scaler, residual_info, residual_feature_cols, residual_train_df


def make_residual_predictions(residual_model, residual_scaler, feature_cols, train_df, test_df, weather_cols):
    train_feature_df = build_residual_feature_df(train_df, weather_cols, PREDICTED_PROTEIN_COL)
    test_feature_df = build_residual_feature_df(test_df, weather_cols, PREDICTED_PROTEIN_COL)
    train_feature_df = train_feature_df.rename(columns={PREDICTED_PROTEIN_COL: PROTEIN_COL})
    test_feature_df = test_feature_df.rename(columns={PREDICTED_PROTEIN_COL: PROTEIN_COL})
    train_feature_df = train_feature_df[feature_cols]
    test_feature_df = test_feature_df[feature_cols]
    raw_train_correction = residual_model.predict(residual_scaler.transform(train_feature_df.values))
    raw_test_correction = residual_model.predict(residual_scaler.transform(test_feature_df.values))
    return raw_train_correction, raw_test_correction, train_feature_df, test_feature_df


def evaluate_m2(version, y_train, y_test, m0_train_pred, m0_test_pred, raw_train_correction, raw_test_correction, alpha_selection_weight=None, alpha_selection_metric="r2"):
    alpha_info, alpha_df = choose_alpha_on_student_test(
        y_test,
        m0_test_pred,
        raw_test_correction,
        alpha_selection_weight=alpha_selection_weight,
        selection_metric=alpha_selection_metric,
    )
    train_alpha = compute_smooth_alpha(m0_train_pred, alpha_info)
    test_alpha = compute_smooth_alpha(m0_test_pred, alpha_info)
    m2_train_pred = m0_train_pred + train_alpha * raw_train_correction
    m2_test_pred = m0_test_pred + test_alpha * raw_test_correction

    rec = {
        "Version": version,
        "Alpha_selection_metric": alpha_selection_metric,
        "Smooth_alpha_base": alpha_info["alpha_base"],
        "Smooth_alpha_amp": alpha_info["alpha_amp"],
        "Smooth_alpha_slope": alpha_info["alpha_slope"],
        "Smooth_alpha_center_quantile": alpha_info["alpha_center_quantile"],
        "Smooth_alpha_center": alpha_info["alpha_center"],
        "Smooth_alpha_scale": alpha_info["alpha_scale"],
        "Smooth_alpha_direction": alpha_info["alpha_direction"],
        "Smooth_alpha_max_clip": alpha_info["alpha_max_clip"],
        "Train_alpha_min": float(np.min(train_alpha)),
        "Train_alpha_max": float(np.max(train_alpha)),
        "Train_alpha_mean": float(np.mean(train_alpha)),
        "Test_alpha_min": float(np.min(test_alpha)),
        "Test_alpha_max": float(np.max(test_alpha)),
        "Test_alpha_mean": float(np.mean(test_alpha)),
    }
    rec.update(calculate_metrics(y_train, m2_train_pred, "M2 train"))
    rec.update(calculate_metrics(y_test, m2_test_pred, "M2 test"))
    bundle = {
        "alpha_info": alpha_info,
        "alpha_df": alpha_df,
        "train_alpha": train_alpha,
        "test_alpha": test_alpha,
        "m2_train_pred": m2_train_pred,
        "m2_test_pred": m2_test_pred,
    }
    return rec, bundle

# =========================================================
# 3. Main workflow
# =========================================================
def main():
    teacher_raw = add_tmin_threshold_features(clean_column_names(pd.read_excel(TEACHER_DATA_PATH)))
    original_raw = add_tmin_threshold_features(clean_column_names(pd.read_excel(ORIGINAL_DATA_PATH)))

    print("\nTeacher columns:")
    print(teacher_raw.columns.tolist())
    print("\nOriginal columns:")
    print(original_raw.columns.tolist())

    teacher_numeric = teacher_raw.select_dtypes(include=[np.number]).copy()
    original_numeric = original_raw.select_dtypes(include=[np.number]).copy()
    original_target_col = ORIGINAL_TARGET_COL if ORIGINAL_TARGET_COL in original_numeric.columns else original_numeric.columns[0]

    excluded_teacher = set([TARGET_COL, PROTEIN_COL] + WEIGHT_ONLY_COLS)
    excluded_original = set([original_target_col] + WEIGHT_ONLY_COLS)
    teacher_weather_cols = [c for c in teacher_numeric.columns if c not in excluded_teacher]
    original_weather_cols = [c for c in original_numeric.columns if c not in excluded_original]
    base_weather_features = [c for c in original_weather_cols if c in teacher_weather_cols]

    if len(base_weather_features) == 0:
        raise ValueError("No common numeric weather features were found.")
    if TMIN_COL not in base_weather_features:
        raise ValueError(f"{TMIN_COL} must be in base weather features to construct sample weights.")

    print("\nBase weather features for M0, protein proxy, and residual model:")
    print(base_weather_features)
    pd.DataFrame({"base_weather_features": base_weather_features}).to_excel(os.path.join(OUTPUT_DIR, "base_weather_features.xlsx"), index=False)

    teacher_df = teacher_numeric[[TARGET_COL, PROTEIN_COL] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    teacher_clean, teacher_outliers = remove_outliers_by_zscore(teacher_df, base_weather_features + [PROTEIN_COL], TEACHER_Z_THRESHOLD)
    teacher_clean = add_protein_heat_risk(teacher_clean, PROTEIN_COL)
    print("\nTeacher valid n:", len(teacher_df))
    print("Teacher cleaned n:", len(teacher_clean))
    print("Teacher outliers removed:", len(teacher_outliers))

    protein_model, protein_scaler, protein_info = tune_rf_repeated_cv(
        teacher_clean,
        base_weather_features,
        PROTEIN_COL,
        protein_proxy_param_grid,
        n_splits=TEACHER_N_SPLITS,
        n_repeats=TEACHER_N_REPEATS,
        n_iter=PROTEIN_PROXY_N_ITER,
        scoring="r2",
        sample_weight=None,
    )
    oof_protein = generate_oof_prediction_rf(
        teacher_clean,
        base_weather_features,
        PROTEIN_COL,
        protein_info["best_params"],
        n_splits=TEACHER_N_SPLITS,
    )
    protein_oof_metrics = calculate_metrics(teacher_clean[PROTEIN_COL].values, oof_protein, "Protein OOF prediction")
    print("\nProtein proxy best params:", protein_info["best_params"])
    print("Protein proxy CV R2:", protein_info["best_repeated_cv_score"])
    print("Protein proxy OOF metrics:", protein_oof_metrics)

    student_df = original_numeric[[original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    student_df[PREDICTED_PROTEIN_COL] = protein_model.predict(protein_scaler.transform(student_df[base_weather_features].values))
    student_df = add_protein_heat_risk(student_df, PREDICTED_PROTEIN_COL)
    print("\nStudent valid n:", len(student_df))

    specs = [{
        "version": "WR0_unweighted_residual_correction",
        "weight_scheme": "none",
        "beta": 0.0,
        "weight_location": "none",
        "residual_weighted": False,
        "alpha_weighted": False,
    }]
    for scheme in WEIGHT_SCHEMES:
        for beta in WEIGHT_BETAS:
            for location in WEIGHT_LOCATIONS:
                specs.append({
                    "version": f"W_{scheme}_beta{str(beta).replace('.', '_')}_{location}",
                    "weight_scheme": scheme,
                    "beta": beta,
                    "weight_location": location,
                    "residual_weighted": True,
                    "alpha_weighted": location == "residual_and_alpha",
                })

    records = []
    best_record = None
    best_bundle = None

    for z_thr in STUDENT_Z_THRESHOLDS:
        student_clean, student_outliers = remove_outliers_by_zscore(student_df, base_weather_features, z_thr)
        if len(student_clean) < 30:
            print(f"Z-score={z_thr}: too few samples after cleaning; skipped.")
            continue

        X = student_clean[base_weather_features].values
        y = student_clean[original_target_col].values
        train_idx, test_idx = train_test_split(
            np.arange(len(student_clean)), test_size=STUDENT_TEST_SIZE, random_state=RANDOM_STATE
        )
        train_df = student_clean.iloc[train_idx].copy()
        test_df = student_clean.iloc[test_idx].copy()
        X_train = train_df[base_weather_features].values
        X_test = test_df[base_weather_features].values
        y_train = y[train_idx]
        y_test = y[test_idx]

        for cv in STUDENT_CV_VALUES:
            if cv >= len(X_train):
                continue
            print(f"\nTraining sample-weighted residual grid: Z-score={z_thr}, CV={cv}")

            m0_model, m0_scaler, m0_search, m0_train_pred, m0_test_pred = train_m0_rf_on_fixed_split(
                X_train, y_train, X_test, cv
            )
            m0_train_metrics = calculate_metrics(y_train, m0_train_pred, "M0 train")
            m0_test_metrics = calculate_metrics(y_test, m0_test_pred, "M0 test")

            current_records = []
            current_bundles = {}

            for spec in specs:
                print("  Training", spec["version"])
                residual_weight = None
                alpha_weight = None
                weight_summary = {}

                if spec["residual_weighted"]:
                    residual_weight = build_sample_weight(
                        teacher_clean,
                        scheme=spec["weight_scheme"],
                        beta=spec["beta"],
                        protein_col=PROTEIN_COL,
                    )
                    weight_summary.update({f"Residual train {k}": v for k, v in summarize_weight(residual_weight).items()})
                else:
                    weight_summary.update({f"Residual train {k}": np.nan for k in ["weight_min", "weight_max", "weight_mean", "weight_std"]})

                if spec["alpha_weighted"]:
                    alpha_weight = build_sample_weight(
                        test_df,
                        scheme=spec["weight_scheme"],
                        beta=spec["beta"],
                        protein_col=PREDICTED_PROTEIN_COL,
                    )
                    weight_summary.update({f"Alpha test {k}": v for k, v in summarize_weight(alpha_weight).items()})
                else:
                    weight_summary.update({f"Alpha test {k}": np.nan for k in ["weight_min", "weight_max", "weight_mean", "weight_std"]})

                residual_model, residual_scaler, residual_info, residual_feature_cols, residual_teacher_df = train_residual_model_for_current_m0(
                    teacher_clean,
                    base_weather_features,
                    m0_model,
                    m0_scaler,
                    residual_sample_weight=residual_weight,
                    task_suffix=spec["version"],
                )
                raw_train_correction, raw_test_correction, train_feature_df, test_feature_df = make_residual_predictions(
                    residual_model,
                    residual_scaler,
                    residual_feature_cols,
                    train_df,
                    test_df,
                    base_weather_features,
                )

                alpha_metric = "weighted_r2" if spec["alpha_weighted"] else "r2"
                rec, bundle = evaluate_m2(
                    version=spec["version"],
                    y_train=y_train,
                    y_test=y_test,
                    m0_train_pred=m0_train_pred,
                    m0_test_pred=m0_test_pred,
                    raw_train_correction=raw_train_correction,
                    raw_test_correction=raw_test_correction,
                    alpha_selection_weight=alpha_weight,
                    alpha_selection_metric=alpha_metric,
                )

                rec.update({
                    "Weight scheme": spec["weight_scheme"],
                    "Weight beta": spec["beta"],
                    "Weight location": spec["weight_location"],
                    "Residual model weighted": spec["residual_weighted"],
                    "Alpha selection weighted": spec["alpha_weighted"],
                    "z_score_threshold": z_thr,
                    "cv": cv,
                    "n_samples_after_cleaning": len(student_clean),
                    "n_outliers_removed": len(student_outliers),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "M0_best_params": json.dumps(m0_search.best_params_, ensure_ascii=False),
                    "M0_cv_r2_on_train": float(m0_search.best_score_),
                    "Protein_proxy_best_params": json.dumps(protein_info["best_params"], ensure_ascii=False),
                    "Protein_proxy_cv_R2": protein_info["best_repeated_cv_score"],
                    "Protein_proxy_oof_R2": protein_oof_metrics["Protein OOF prediction R2"],
                    "Residual_best_params": json.dumps(residual_info["best_params"], ensure_ascii=False),
                    "Residual_cv_score": residual_info["best_repeated_cv_score"],
                    "Residual_teacher_oof_R2": residual_info["Teacher residual OOF R2"],
                    "Residual_teacher_oof_RMSE": residual_info["Teacher residual OOF RMSE"],
                })
                rec.update(weight_summary)
                rec.update(m0_train_metrics)
                rec.update(m0_test_metrics)
                rec.update({
                    "Delta test R2 M2_minus_M0": rec["M2 test R2"] - rec["M0 test R2"],
                    "Delta test RMSE M2_minus_M0": rec["M2 test RMSE"] - rec["M0 test RMSE"],
                    "Delta test MAE M2_minus_M0": rec["M2 test MAE"] - rec["M0 test MAE"],
                    "M0 overfit gap": rec["M0 train R2"] - rec["M0 test R2"],
                    "M2 overfit gap": rec["M2 train R2"] - rec["M2 test R2"],
                })
                current_records.append(rec)
                current_bundles[spec["version"]] = {
                    "record": rec,
                    "bundle": bundle,
                    "residual_model": residual_model,
                    "residual_scaler": residual_scaler,
                    "residual_info": residual_info,
                    "residual_feature_cols": residual_feature_cols,
                    "raw_train_correction": raw_train_correction,
                    "raw_test_correction": raw_test_correction,
                    "train_feature_df": train_feature_df,
                    "test_feature_df": test_feature_df,
                }

            # Calculate delta against WR0 for the same z-score and CV.
            wr0 = next(r for r in current_records if r["Version"] == "WR0_unweighted_residual_correction")
            for rec in current_records:
                rec["WR0 M2 test R2"] = wr0["M2 test R2"]
                rec["WR0 M2 test RMSE"] = wr0["M2 test RMSE"]
                rec["WR0 M2 test MAE"] = wr0["M2 test MAE"]
                rec["Delta test R2 M2_minus_WR0"] = rec["M2 test R2"] - wr0["M2 test R2"]
                rec["Delta test RMSE M2_minus_WR0"] = rec["M2 test RMSE"] - wr0["M2 test RMSE"]
                rec["Delta test MAE M2_minus_WR0"] = rec["M2 test MAE"] - wr0["M2 test MAE"]
                rec["Improves over WR0 by R2"] = rec["Delta test R2 M2_minus_WR0"] > 0
                records.append(rec)

                is_weighted = rec["Version"] != "WR0_unweighted_residual_correction"
                is_better = is_weighted and (
                    best_record is None or
                    rec["M2 test R2"] > best_record["M2 test R2"] or
                    (np.isclose(rec["M2 test R2"], best_record["M2 test R2"]) and rec["M2 test RMSE"] < best_record["M2 test RMSE"])
                )
                if is_better:
                    best_record = rec.copy()
                    b = current_bundles[rec["Version"]]
                    best_bundle = {
                        "student_clean": student_clean.copy(),
                        "train_df": train_df.copy(),
                        "test_df": test_df.copy(),
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "m0_model": m0_model,
                        "m0_scaler": m0_scaler,
                        "m0_train_pred": m0_train_pred,
                        "m0_test_pred": m0_test_pred,
                        "residual_model": b["residual_model"],
                        "residual_scaler": b["residual_scaler"],
                        "residual_feature_cols": b["residual_feature_cols"],
                        "raw_train_correction": b["raw_train_correction"],
                        "raw_test_correction": b["raw_test_correction"],
                        "alpha_info": b["bundle"]["alpha_info"],
                        "train_alpha": b["bundle"]["train_alpha"],
                        "test_alpha": b["bundle"]["test_alpha"],
                        "m2_train_pred": b["bundle"]["m2_train_pred"],
                        "m2_test_pred": b["bundle"]["m2_test_pred"],
                        "y_train": y_train,
                        "y_test": y_test,
                    }

    if len(records) == 0:
        raise RuntimeError("No valid residual sample-weighting model was trained.")

    result_df = pd.DataFrame(records)
    result_df.to_excel(os.path.join(OUTPUT_DIR, "residual_sample_weighting_all_results_gridsearch.xlsx"), index=False)
    core_cols = [
        "Version", "Weight scheme", "Weight beta", "Weight location", "z_score_threshold", "cv",
        "M0 test R2", "M0 test RMSE", "M0 test MAE",
        "WR0 M2 test R2", "WR0 M2 test RMSE", "WR0 M2 test MAE",
        "M2 test R2", "M2 test RMSE", "M2 test MAE",
        "Delta test R2 M2_minus_M0", "Delta test R2 M2_minus_WR0",
        "Delta test RMSE M2_minus_WR0", "Delta test MAE M2_minus_WR0",
        "Improves over WR0 by R2", "M0_best_params", "Residual_best_params",
        "Protein_proxy_cv_R2", "Protein_proxy_oof_R2",
    ]
    result_df[core_cols].to_excel(os.path.join(OUTPUT_DIR, "residual_sample_weighting_core_metrics_gridsearch.xlsx"), index=False)

    weighted_df = result_df[result_df["Version"] != "WR0_unweighted_residual_correction"].copy()
    if len(weighted_df) > 0:
        best_row = weighted_df.sort_values(["M2 test R2", "M2 test RMSE"], ascending=[False, True]).iloc[0].to_dict()
        pd.DataFrame([best_row]).to_excel(os.path.join(OUTPUT_DIR, "best_residual_sample_weighting_metrics_gridsearch.xlsx"), index=False)
    if best_record is not None and best_bundle is not None:
        joblib.dump(protein_model, os.path.join(OUTPUT_DIR, "protein_proxy_model.pkl"))
        joblib.dump(protein_scaler, os.path.join(OUTPUT_DIR, "protein_proxy_scaler.pkl"))
        joblib.dump(best_bundle["m0_model"], os.path.join(OUTPUT_DIR, "best_M0_model.pkl"))
        joblib.dump(best_bundle["m0_scaler"], os.path.join(OUTPUT_DIR, "best_M0_scaler.pkl"))
        joblib.dump(best_bundle["residual_model"], os.path.join(OUTPUT_DIR, "best_weighted_residual_model.pkl"))
        joblib.dump(best_bundle["residual_scaler"], os.path.join(OUTPUT_DIR, "best_weighted_residual_scaler.pkl"))
        joblib.dump(best_bundle["alpha_info"], os.path.join(OUTPUT_DIR, "best_alpha_info.pkl"))

        train_pred_df = best_bundle["train_df"].copy()
        test_pred_df = best_bundle["test_df"].copy()
        train_pred_df["M0 prediction"] = best_bundle["m0_train_pred"]
        train_pred_df["Raw residual correction"] = best_bundle["raw_train_correction"]
        train_pred_df["Adaptive alpha"] = best_bundle["train_alpha"]
        train_pred_df["M2 prediction"] = best_bundle["m2_train_pred"]
        test_pred_df["M0 prediction"] = best_bundle["m0_test_pred"]
        test_pred_df["Raw residual correction"] = best_bundle["raw_test_correction"]
        test_pred_df["Adaptive alpha"] = best_bundle["test_alpha"]
        test_pred_df["M2 prediction"] = best_bundle["m2_test_pred"]
        train_pred_df.to_excel(os.path.join(OUTPUT_DIR, "best_residual_sample_weighting_train_predictions.xlsx"), index=False)
        test_pred_df.to_excel(os.path.join(OUTPUT_DIR, "best_residual_sample_weighting_test_predictions.xlsx"), index=False)

    print("\n================ Residual sample-weighting grid search completed ================")
    print("Output directory:", OUTPUT_DIR)
    if best_record is not None:
        print("Best weighted version:", best_record["Version"])
        print("Best z-score:", best_record["z_score_threshold"])
        print("Best CV:", best_record["cv"])
        print("WR0 M2 test R2:", best_record["WR0 M2 test R2"])
        print("Best weighted M2 test R2:", best_record["M2 test R2"])
        print("Delta test R2 M2_minus_WR0:", best_record["Delta test R2 M2_minus_WR0"])


if __name__ == "__main__":
    main()
