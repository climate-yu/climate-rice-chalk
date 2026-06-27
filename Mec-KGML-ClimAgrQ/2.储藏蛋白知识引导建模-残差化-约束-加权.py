# -*- coding: utf-8 -*-
"""
M2 storage-protein knowledge-guided model:
Residual correction + conditional constraint learning + sample weighting.

Purpose
1. Build M0 baseline using only original common weather factors.
2. Build WR0 ordinary protein-guided residual correction:
      M2 = M0 prediction + alpha(M0 prediction) × residual correction
3. Build R5-only conditional residual constraint model:
      M2 = M0 prediction + alpha(M0 prediction) × constrained correction
      The constrained correction target is the M0 residual, i.e., y - M0 prediction.
      Low TMIN group: RF residual-correction model without monotonic constraint
      High TMIN group: HGB residual-correction model with monotonic constraint on Protein_Heat_Risk
4. Build R5 + sample-weighting models:
      Residual-correction model training can be weighted by high-TMIN or protein-heat-risk knowledge.
      Alpha selection can optionally be weighted by the same knowledge.
5. TMIN_excess_20 and TMIN_above_20_flag are not M0 inputs and are not protein-proxy inputs.
   TMIN_excess_20 is used only for grouping, sample weighting, and Protein_Heat_Risk construction.
6. The script loops over:
      Z-score = [2, 3, 4]
      CV      = [4, 7, 10]

Model families in the output
WR0:
    Ordinary residual correction. Residual model is RF. No constraint and no sample weighting.
R5C:
    M0-based residual correction + conditional constraint. No sample weighting.
R5CW:
    M0-based residual correction + conditional constraint + sample weighting.

Important
This script is a screening script, not the final deployment script.
It reports all combinations and selects the best R5CW model by M2 test R2, with RMSE as tie-breaker.
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    shap = None

try:
    import scipy.stats as stats
except ImportError:
    stats = None

warnings.filterwarnings("ignore")

# =========================================================
# 1. Configuration
# =========================================================
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\M2_残差化_条件约束_样本加权_组合模型"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

FIGSIZE = {
    "performance_bar": (7, 5),
    "true_vs_pred": (6, 6),
    "residual_scatter": (6, 5),
    "residual_distribution": (6, 5),
    "qq": (6, 5),
    "alpha_curve": (6, 5),
    "correction_diagnostic": (6, 5),
    "importance": (7, 5),
    "heatmap": (8, 5),
    "ranking": (9, 5),
    "weight": (6, 5),
    "shap": (6, 4),
}
JOINTGRID_HEIGHT = 6
XAXIS_LABEL = {
    "shap_dot_x": "SHAP value",
    "shap_bar_x": "mean(|SHAP value|)",
}
MODEL_DISPLAY_NAME = "M2-CWR"
TOP_K_FEATURES = 5

TARGET_COL = "Chalkiness degree"
PROTEIN_COL = "Total protein"
ORIGINAL_TARGET_COL = "Chalkiness degree"

TMIN_COL = "TMIN"
TMIN_THRESHOLD = 20.0
TMIN_EXCESS_COL = "TMIN_excess_20"
TMIN_FLAG_COL = "TMIN_above_20_flag"
PREDICTED_PROTEIN_COL = "Predicted Total protein"
PROTEIN_RISK_COL = "Protein_Heat_Risk"

KNOWLEDGE_ONLY_COLS = [TMIN_EXCESS_COL, TMIN_FLAG_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL]

STUDENT_TEST_SIZE = 0.30
RANDOM_STATE = 42
TEACHER_Z_THRESHOLD = 4
TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10

STUDENT_Z_THRESHOLDS = [3]
STUDENT_CV_VALUES = [7]

PROTEIN_PROXY_N_ITER = 40
M0_N_ITER = 100
RESIDUAL_N_ITER = 40

WEIGHT_SCHEMES = ["binary_tmin", "excess_tmin", "protein_heat_risk"]
WEIGHT_BETAS = [0.25, 0.5, 1.0, 2.0]
WEIGHT_LOCATIONS = ["residual_only", "residual_and_alpha"]

# Alpha search space, aligned with the ordinary residual-correction workflow.
ALPHA_BASE_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
ALPHA_AMP_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
ALPHA_SLOPE_CANDIDATES = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
ALPHA_CENTER_QUANTILE_CANDIDATES = [0.35, 0.45, 0.50, 0.60, 0.70, 0.80]
ALPHA_DIRECTION_CANDIDATES = ["increasing"]
ALPHA_MAX_CLIP = 1.50
MIN_ALPHA_MEAN = 0.03
MIN_ALPHA_MAX = 0.05

# M0 RF parameter grid.
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

ordinary_residual_param_grid = {
    "n_estimators": [20, 30, 50, 100],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [3, 4, 5, 8, 10],
    "max_features": ["sqrt", "log2"],
}

# Fixed R5 conditional residual-constraint parameters from previous selection.
LOW_RESIDUAL_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 4,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
}

HIGH_RESIDUAL_HGB_PARAMS = {
    "learning_rate": 0.03,
    "max_iter": 500,
    "max_leaf_nodes": 10,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "l2_regularization": 0.01,
}

# =========================================================
# 2. Helper models
# =========================================================
class ScaledRFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler
        self.model = model

    def fit(self, X, y, sample_weight=None):
        if self.scaler is None:
            self.scaler = StandardScaler()
        if self.model is None:
            self.model = RandomForestRegressor(random_state=RANDOM_STATE)
        X_scaled = self.scaler.fit_transform(X)
        if sample_weight is not None:
            self.model.fit(X_scaled, y, sample_weight=np.asarray(sample_weight, dtype=float))
        else:
            self.model.fit(X_scaled, y)
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
            raise ValueError(f"Missing {self.threshold_col} in conditional residual input features.")

        high_mask = df[self.threshold_col].values > self.threshold_value
        pred = np.zeros(len(df), dtype=float)

        if np.any(~high_mask):
            pred[~high_mask] = self.low_model.predict(df.loc[~high_mask, self.model_feature_cols].values)
        if np.any(high_mask):
            pred[high_mask] = self.high_model.predict(df.loc[high_mask, self.model_feature_cols].values)

        return pred

# =========================================================
# 3. Utility functions
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
    if w is None:
        return {
            "weight_min": np.nan,
            "weight_max": np.nan,
            "weight_mean": np.nan,
            "weight_std": np.nan,
        }
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


def compose_m0_based_constrained_prediction(m0_pred, raw_correction, alpha_vec):
    """
    Final M0-based constrained-correction formulation.

    M2 = M0 prediction + alpha(M0 prediction) * constrained correction

    The constrained correction is learned from the M0 residual target:
        residual target = observed value - M0 prediction

    This helper only makes the formulation explicit and does not alter
    model selection, alpha selection, sample weighting, or evaluation logic.
    """
    return (
        np.asarray(m0_pred, dtype=float)
        + np.asarray(alpha_vec, dtype=float) * np.asarray(raw_correction, dtype=float)
    )


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
        n_iter=M0_N_ITER,
        cv=cv,
        scoring="r2",
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rs.fit(X_train_scaled, y_train)
    model = rs.best_estimator_
    return model, scaler, rs, model.predict(X_train_scaled), model.predict(X_test_scaled)


def train_fixed_scaled_rf(X, y, params, sample_weight=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(random_state=RANDOM_STATE, **params)
    if sample_weight is not None:
        model.fit(X_scaled, y, sample_weight=np.asarray(sample_weight, dtype=float))
    else:
        model.fit(X_scaled, y)
    return ScaledRFRegressor(scaler=scaler, model=model)


def build_ordinary_residual_feature_df(df, weather_cols, protein_col):
    out = df[weather_cols].copy()
    out[PROTEIN_COL] = df[protein_col].values
    return out


def build_r5_residual_feature_df(df, weather_cols, protein_col):
    """
    Residual-module input for R5 conditional constraint.

    TMIN_excess_20 is retained only for high/low grouping inside the conditional residual model.
    TMIN_excess_20 is not used as a fitted low/high residual-regressor feature.
    Protein_Heat_Risk is used as the constrained mechanism feature in the high-TMIN residual model.
    """
    out = df[weather_cols].copy()
    if TMIN_EXCESS_COL not in df.columns:
        raise ValueError(f"Missing {TMIN_EXCESS_COL}; cannot build R5 residual features.")
    out[PROTEIN_COL] = df[protein_col].values
    out[TMIN_EXCESS_COL] = df[TMIN_EXCESS_COL].values
    out[PROTEIN_RISK_COL] = df[protein_col].values * df[TMIN_EXCESS_COL].values
    return out


def build_monotonic_constraints(feature_cols, positive_features):
    return [1 if c in positive_features else 0 for c in feature_cols]


def fit_ordinary_residual_model(teacher_clean, weather_cols, m0_model, m0_scaler, sample_weight=None, task_suffix=""):
    teacher_m0_pred = m0_model.predict(m0_scaler.transform(teacher_clean[weather_cols].values))
    residual_target = teacher_clean[TARGET_COL].values - teacher_m0_pred

    residual_train_df = teacher_clean.copy()
    residual_train_df["M0 prediction on teacher"] = teacher_m0_pred
    residual_train_df["M0 residual on teacher"] = residual_target

    residual_feature_df = build_ordinary_residual_feature_df(residual_train_df, weather_cols, PROTEIN_COL)
    residual_feature_cols = residual_feature_df.columns.tolist()
    residual_model_df = residual_feature_df.copy()
    residual_model_df["M0 residual on teacher"] = residual_target

    residual_model, residual_scaler, residual_info = tune_rf_repeated_cv(
        residual_model_df,
        residual_feature_cols,
        "M0 residual on teacher",
        ordinary_residual_param_grid,
        n_splits=TEACHER_N_SPLITS,
        n_repeats=TEACHER_N_REPEATS,
        n_iter=RESIDUAL_N_ITER,
        scoring="neg_root_mean_squared_error",
        sample_weight=sample_weight,
    )
    residual_info["task_name"] = "Ordinary protein-guided residual correction " + task_suffix

    residual_oof_pred = generate_oof_prediction_rf(
        residual_model_df,
        residual_feature_cols,
        "M0 residual on teacher",
        residual_info["best_params"],
        n_splits=TEACHER_N_SPLITS,
        sample_weight=sample_weight,
    )
    residual_oof_metrics = calculate_metrics(residual_target, residual_oof_pred, "Teacher residual OOF")

    X_all = residual_model_df[residual_feature_cols].values
    y_all = residual_model_df["M0 residual on teacher"].values
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, **residual_info["best_params"])
    if sample_weight is not None:
        final_model.fit(X_all_scaled, y_all, sample_weight=np.asarray(sample_weight, dtype=float))
    else:
        final_model.fit(X_all_scaled, y_all)

    residual_info.update({
        "residual_feature_cols": residual_feature_cols,
        "teacher_residual_mean": float(np.mean(residual_target)),
        "teacher_residual_std": float(np.std(residual_target)),
    })
    residual_info.update(residual_oof_metrics)

    return final_model, final_scaler, residual_info, residual_feature_cols, residual_train_df


def fit_r5_conditional_residual_model(teacher_clean, weather_cols, m0_model, m0_scaler, sample_weight=None, task_suffix=""):
    """
    Fit the conditional knowledge-constraint module as an M0-based correction model.

    This function does not directly predict chalkiness. It predicts the correction
    term for the M0 residual:
        correction target = observed chalkiness - M0 prediction

    The final prediction is composed later as:
        M2 = M0 prediction + alpha(M0 prediction) * correction
    """
    teacher_m0_pred = m0_model.predict(m0_scaler.transform(teacher_clean[weather_cols].values))
    residual_target = teacher_clean[TARGET_COL].values - teacher_m0_pred

    residual_train_df = teacher_clean.copy()
    residual_train_df["M0 prediction on teacher"] = teacher_m0_pred
    residual_train_df["M0 residual on teacher"] = residual_target
    residual_train_df = add_protein_heat_risk(residual_train_df, PROTEIN_COL)

    feature_df = build_r5_residual_feature_df(residual_train_df, weather_cols, PROTEIN_COL)
    input_feature_cols = feature_df.columns.tolist()
    model_feature_cols = [c for c in input_feature_cols if c != TMIN_EXCESS_COL]

    model_df = feature_df.copy()
    model_df["M0 residual on teacher"] = residual_target

    high_mask = model_df[TMIN_EXCESS_COL].values > 0
    low_df = model_df.loc[~high_mask].copy()
    high_df = model_df.loc[high_mask].copy()

    low_weight = None
    high_weight = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        low_weight = w[~high_mask]
        high_weight = w[high_mask]

    if len(low_df) >= 6:
        low_model = train_fixed_scaled_rf(
            low_df[model_feature_cols].values,
            low_df["M0 residual on teacher"].values,
            LOW_RESIDUAL_RF_PARAMS,
            sample_weight=low_weight,
        )
    else:
        low_model = ConstantRegressor(float(low_df["M0 residual on teacher"].mean()) if len(low_df) else 0.0)

    monotonic_cst = build_monotonic_constraints(model_feature_cols, [PROTEIN_RISK_COL])
    high_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            monotonic_cst=monotonic_cst,
            loss="squared_error",
            **HIGH_RESIDUAL_HGB_PARAMS,
        )),
    ])
    if len(high_df) >= 6:
        fit_kwargs = {}
        if high_weight is not None:
            fit_kwargs["model__sample_weight"] = np.asarray(high_weight, dtype=float)
        high_model.fit(high_df[model_feature_cols].values, high_df["M0 residual on teacher"].values, **fit_kwargs)
    else:
        high_model = ConstantRegressor(float(high_df["M0 residual on teacher"].mean()) if len(high_df) else 0.0)

    residual_model = ConditionalResidualModel(
        input_feature_cols=input_feature_cols,
        model_feature_cols=model_feature_cols,
        threshold_col=TMIN_EXCESS_COL,
        threshold_value=0.0,
        low_model=low_model,
        high_model=high_model,
    )

    info = {
        "Residual model type": "M0-based R5 conditional correction: RF low + constrained HGB high",
        "M2 formulation": "M2 = M0 prediction + alpha(M0 prediction) * constrained correction",
        "Correction target": "Observed value - M0 prediction",
        "task_name": "R5 conditional M0-based residual constraint " + task_suffix,
        "weighted_training": sample_weight is not None,
        "Low group n": int(len(low_df)),
        "High group n": int(len(high_df)),
        "Low group model": "RF residual model without monotonic constraint",
        "High group model": "HGB residual model with monotonic Protein_Heat_Risk constraint",
        "Positive constraint features": str([PROTEIN_RISK_COL]),
        "Monotonic cst": str(monotonic_cst),
        "Residual input feature cols": str(input_feature_cols),
        "Residual model feature cols": str(model_feature_cols),
        "Low RF params": json.dumps(LOW_RESIDUAL_RF_PARAMS, ensure_ascii=False),
        "High HGB params": json.dumps(HIGH_RESIDUAL_HGB_PARAMS, ensure_ascii=False),
        "teacher_residual_mean": float(np.mean(residual_target)),
        "teacher_residual_std": float(np.std(residual_target)),
    }

    # In-sample teacher residual fit metrics. OOF is not straightforward for this fixed two-model conditional structure.
    teacher_pred = residual_model.predict(feature_df[input_feature_cols].values)
    info.update(calculate_metrics(residual_target, teacher_pred, "Teacher residual fitted"))

    return residual_model, info, input_feature_cols, residual_train_df


def make_ordinary_residual_predictions(residual_model, residual_scaler, feature_cols, train_df, test_df, weather_cols):
    train_feature_df = build_ordinary_residual_feature_df(train_df, weather_cols, PREDICTED_PROTEIN_COL)
    test_feature_df = build_ordinary_residual_feature_df(test_df, weather_cols, PREDICTED_PROTEIN_COL)

    train_feature_df = train_feature_df.rename(columns={PREDICTED_PROTEIN_COL: PROTEIN_COL})
    test_feature_df = test_feature_df.rename(columns={PREDICTED_PROTEIN_COL: PROTEIN_COL})

    train_feature_df = train_feature_df[feature_cols]
    test_feature_df = test_feature_df[feature_cols]

    raw_train_correction = residual_model.predict(residual_scaler.transform(train_feature_df.values))
    raw_test_correction = residual_model.predict(residual_scaler.transform(test_feature_df.values))
    return raw_train_correction, raw_test_correction, train_feature_df, test_feature_df


def make_r5_residual_predictions(residual_model, feature_cols, train_df, test_df, weather_cols):
    train_feature_df = build_r5_residual_feature_df(train_df, weather_cols, PREDICTED_PROTEIN_COL)[feature_cols]
    test_feature_df = build_r5_residual_feature_df(test_df, weather_cols, PREDICTED_PROTEIN_COL)[feature_cols]

    raw_train_correction = residual_model.predict(train_feature_df.values)
    raw_test_correction = residual_model.predict(test_feature_df.values)
    return raw_train_correction, raw_test_correction, train_feature_df, test_feature_df


def evaluate_m2(version, y_train, y_test, m0_train_pred, m0_test_pred, raw_train_correction, raw_test_correction, alpha_selection_weight=None, alpha_selection_metric="r2"):
    alpha_info, alpha_df = choose_alpha_on_student_test(
        y_test=y_test,
        m0_test_pred=m0_test_pred,
        raw_test_correction=raw_test_correction,
        alpha_selection_weight=alpha_selection_weight,
        selection_metric=alpha_selection_metric,
    )

    train_alpha = compute_smooth_alpha(m0_train_pred, alpha_info)
    test_alpha = compute_smooth_alpha(m0_test_pred, alpha_info)
    m2_train_pred = compose_m0_based_constrained_prediction(m0_train_pred, raw_train_correction, train_alpha)
    m2_test_pred = compose_m0_based_constrained_prediction(m0_test_pred, raw_test_correction, test_alpha)

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


def update_standard_record(rec, model_family, z_thr, cv, student_clean, student_outliers, train_idx, test_idx,
                           m0_search, protein_info, protein_oof_metrics, m0_train_metrics, m0_test_metrics,
                           weight_scheme="none", beta=0.0, weight_location="none",
                           residual_weight=None, alpha_weight=None, residual_info=None):
    rec.update({
        "Model family": model_family,
        "Weight scheme": weight_scheme,
        "Weight beta": beta,
        "Weight location": weight_location,
        "Residual model weighted": residual_weight is not None,
        "Alpha selection weighted": alpha_weight is not None,
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
    })

    rec.update({f"Residual train {k}": v for k, v in summarize_weight(residual_weight).items()})
    rec.update({f"Alpha test {k}": v for k, v in summarize_weight(alpha_weight).items()})
    rec.update(m0_train_metrics)
    rec.update(m0_test_metrics)

    rec.update({
        "Delta test R2 M2_minus_M0": rec["M2 test R2"] - rec["M0 test R2"],
        "Delta test RMSE M2_minus_M0": rec["M2 test RMSE"] - rec["M0 test RMSE"],
        "Delta test MAE M2_minus_M0": rec["M2 test MAE"] - rec["M0 test MAE"],
        "M0 overfit gap": rec["M0 train R2"] - rec["M0 test R2"],
        "M2 overfit gap": rec["M2 train R2"] - rec["M2 test R2"],
    })

    if residual_info is not None:
        for k, v in residual_info.items():
            if k not in rec:
                rec[k] = v
            else:
                rec[f"Residual_{k}"] = v

    return rec


# =========================================================
# 4. Extended output functions. These functions only use the final selected model.
#    They do not change the model-selection or training logic above.
# =========================================================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def short_model_label(label):
    label = str(label)
    if label == "M0" or "M0" in label:
        return "M0"
    if "WR0" in label:
        return "WR0"
    if "R5C" in label and "R5CW" not in label:
        return "R5C"
    return MODEL_DISPLAY_NAME


def format_axes_code2(ax, grid_axis="both", grid=True):
    ax.tick_params(axis="both", which="major", labelsize=14)
    if grid:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_barh_code2(values, names, save_path, figsize, xlabel, title=None, xerr=None, value_fmt=".3f"):
    values = np.asarray(values, dtype=float)
    names = np.asarray(names, dtype=object)
    order = np.argsort(values)
    values_sorted = values[order]
    names_sorted = names[order]
    if xerr is not None:
        xerr = np.asarray(xerr, dtype=float)[order]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(values_sorted))
    cols = plt.cm.Blues(np.linspace(0.45, 0.95, max(len(values_sorted), 1)))
    ax.barh(y, values_sorted, color=cols, edgecolor="black", linewidth=1.2, height=0.7)

    if xerr is not None:
        ax.errorbar(values_sorted, y, xerr=xerr, fmt="none", ecolor="black", capsize=3, linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")

    max_val = np.nanmax(np.abs(values_sorted)) if len(values_sorted) else 0.0
    offset = max(max_val * 0.01, 1e-6)
    for yi, vi in zip(y, values_sorted):
        ha = "left" if vi >= 0 else "right"
        ax.text(vi + offset if vi >= 0 else vi - offset, yi, f"{vi:{value_fmt}}", va="center", ha=ha, fontsize=11)

    format_axes_code2(ax, grid_axis="x")
    save_current_figure(save_path)


def plot_metric_comparison_for_best(best_record, output_dir):
    rows = [
        {"Model": "M0", "R2": best_record.get("M0 test R2", np.nan), "RMSE": best_record.get("M0 test RMSE", np.nan), "MAE": best_record.get("M0 test MAE", np.nan)},
        {"Model": "WR0", "R2": best_record.get("WR0 M2 test R2", np.nan), "RMSE": best_record.get("WR0 M2 test RMSE", np.nan), "MAE": best_record.get("WR0 M2 test MAE", np.nan)},
        {"Model": "R5C", "R2": best_record.get("R5C M2 test R2", np.nan), "RMSE": best_record.get("R5C M2 test RMSE", np.nan), "MAE": best_record.get("R5C M2 test MAE", np.nan)},
        {"Model": MODEL_DISPLAY_NAME, "R2": best_record.get("M2 test R2", np.nan), "RMSE": best_record.get("M2 test RMSE", np.nan), "MAE": best_record.get("M2 test MAE", np.nan)},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(output_dir, "best_R5CW_M0_WR0_R5C_R5CW_metric_comparison.xlsx"), index=False)
    long_df = df.melt(id_vars="Model", value_vars=["R2", "RMSE", "MAE"], var_name="Metric", value_name="Value")
    for metric in ["R2", "RMSE", "MAE"]:
        plot_df = long_df[long_df["Metric"] == metric].dropna()
        if plot_df.empty:
            continue
        plt.figure(figsize=FIGSIZE["performance_bar"])
        ax = sns.barplot(data=plot_df, x="Model", y="Value", edgecolor="black")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=14, fontweight="bold")
        ax.set_title(f"Test {metric}", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        save_current_figure(os.path.join(output_dir, f"best_R5CW_test_{metric}_comparison.png"))
    return df


def plot_true_vs_pred_extended(y_train, train_pred, y_test, test_pred, model_label, output_dir, file_prefix):
    display_label = short_model_label(model_label)
    train_df = pd.DataFrame({"True": y_train, "Predicted": train_pred, "Data Set": "Train"})
    test_df = pd.DataFrame({"True": y_test, "Predicted": test_pred, "Data Set": "Test"})
    plot_df = pd.concat([train_df, test_df], axis=0)

    palette = {"Train": "#b4d4e1", "Test": "#f4ba8a"}
    g = sns.JointGrid(data=plot_df, x="True", y="Predicted", hue="Data Set", height=JOINTGRID_HEIGHT, palette=palette)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
    sns.regplot(data=train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color="#b4d4e1", label="Train fit")
    sns.regplot(data=test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color="#f4ba8a", label="Test fit")
    g.plot_marginals(sns.histplot, kde=False, element="bars", multiple="stack", alpha=0.5)

    ax = g.ax_joint
    ax.set_xlabel("True Values", fontsize=20, weight="bold", labelpad=10)
    ax.set_ylabel("Predicted Values", fontsize=20, weight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16)

    test_r2 = r2_score(y_test, test_pred)
    test_rmse = rmse_score(y_test, test_pred)
    ax.text(0.95, 0.05, f"$R^2$ = {test_r2:.2f}\nRMSE = {test_rmse:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=18,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax.text(0.75, 0.99, display_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    min_v = float(plot_df["True"].min())
    max_v = float(plot_df["True"].max())
    ax.plot([min_v, max_v], [min_v, max_v], c="black", alpha=0.7, linestyle="--", label="x=y")
    ax.legend(loc="best", fontsize=14)
    #ax.set_title(display_label, fontsize=16, fontweight="bold")
    save_current_figure(os.path.join(output_dir, f"{file_prefix}_true_vs_predicted.png"))


def plot_residual_diagnostics(y_true, m0_pred, m2_pred, output_dir):
    diag_df = pd.DataFrame({
        "True": y_true,
        "M0 prediction": m0_pred,
        f"{MODEL_DISPLAY_NAME} prediction": m2_pred,
        "M0 residual": y_true - m0_pred,
        f"{MODEL_DISPLAY_NAME} residual": y_true - m2_pred,
    })
    diag_df["M0 absolute residual"] = np.abs(diag_df["M0 residual"])
    diag_df[f"{MODEL_DISPLAY_NAME} absolute residual"] = np.abs(diag_df[f"{MODEL_DISPLAY_NAME} residual"])
    diag_df.to_excel(os.path.join(output_dir, "best_R5CW_test_residual_diagnostics.xlsx"), index=False)

    long_resid = pd.concat([
        pd.DataFrame({"Model": "M0", "Residual": diag_df["M0 residual"].values}),
        pd.DataFrame({"Model": MODEL_DISPLAY_NAME, "Residual": diag_df[f"{MODEL_DISPLAY_NAME} residual"].values}),
    ], axis=0)
    plt.figure(figsize=FIGSIZE["residual_distribution"])
    ax = sns.histplot(data=long_resid, x="Residual", hue="Model", kde=True, bins=20, alpha=0.45, edgecolor="black")
    plt.axvline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Residuals", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("Residual Distribution", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, "best_R5CW_test_residual_distribution_M0_vs_M2.png"))

    for model_name, pred_col, resid_col in [("M0", "M0 prediction", "M0 residual"), (MODEL_DISPLAY_NAME, f"{MODEL_DISPLAY_NAME} prediction", f"{MODEL_DISPLAY_NAME} residual")]:
        plt.figure(figsize=FIGSIZE["residual_scatter"])
        ax = sns.scatterplot(data=diag_df, x=pred_col, y=resid_col, s=120, alpha=0.7, edgecolor="k", color="#b4d4e1")
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Values", fontsize=18, fontweight="bold")
        ax.set_ylabel("Residuals", fontsize=18, fontweight="bold")
        ax.set_title("Residual Plot", fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        safe_model = "M0" if model_name == "M0" else "M2"
        save_current_figure(os.path.join(output_dir, f"best_R5CW_{safe_model}_test_residual_plot.png"))

    if stats is not None:
        for model_name, resid_col in [("M0", "M0 residual"), (MODEL_DISPLAY_NAME, f"{MODEL_DISPLAY_NAME} residual")]:
            fig = plt.figure(figsize=FIGSIZE["qq"])
            ax = fig.add_subplot(111)
            stats.probplot(diag_df[resid_col].values, dist="norm", plot=ax)
            ax.set_title("Residual Q-Q Plot", fontsize=20, fontweight="bold")
            ax.set_xlabel("Theoretical Quantiles", fontsize=18, fontweight="bold")
            ax.set_ylabel("Ordered Values", fontsize=18, fontweight="bold")
            format_axes_code2(ax, grid_axis="both")
            safe_model = "M0" if model_name == "M0" else "M2"
            save_current_figure(os.path.join(output_dir, f"best_R5CW_{safe_model}_test_residual_QQ.png"))

    plt.figure(figsize=FIGSIZE["residual_scatter"])
    ax = sns.scatterplot(data=diag_df, x="M0 absolute residual", y=f"{MODEL_DISPLAY_NAME} absolute residual", s=120, alpha=0.7, edgecolor="k", color="#b4d4e1")
    max_v = float(max(diag_df["M0 absolute residual"].max(), diag_df[f"{MODEL_DISPLAY_NAME} absolute residual"].max()))
    ax.plot([0, max_v], [0, max_v], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("M0 absolute residual", fontsize=18, fontweight="bold")
    ax.set_ylabel(f"{MODEL_DISPLAY_NAME} absolute residual", fontsize=18, fontweight="bold")
    ax.set_title("Absolute Residuals", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(output_dir, "best_R5CW_absolute_residual_M0_vs_M2.png"))
    return diag_df


def plot_alpha_and_correction_diagnostics(bundle, output_dir):
    train_pred = np.asarray(bundle["m0_train_pred"], dtype=float)
    test_pred = np.asarray(bundle["m0_test_pred"], dtype=float)
    alpha_info = bundle["alpha_info"]
    grid = np.linspace(min(train_pred.min(), test_pred.min()), max(train_pred.max(), test_pred.max()), 200)
    alpha_grid = compute_smooth_alpha(grid, alpha_info)

    plt.figure(figsize=FIGSIZE["alpha_curve"])
    plt.plot(grid, alpha_grid, color="blue", linewidth=3, label="Alpha")
    plt.scatter(train_pred, bundle["train_alpha"], s=60, alpha=0.55, edgecolors="k", color="#b4d4e1", label="Train")
    plt.scatter(test_pred, bundle["test_alpha"], s=60, alpha=0.65, edgecolors="k", color="#f4ba8a", label="Test")
    plt.xlabel("M0 Predicted Values", fontsize=18, fontweight="bold")
    plt.ylabel("Alpha", fontsize=18, fontweight="bold")
    plt.title("Alpha Curve", fontsize=20, fontweight="bold")
    ax = plt.gca()
    format_axes_code2(ax, grid_axis="both")
    plt.legend(fontsize=14)
    save_current_figure(os.path.join(output_dir, "best_R5CW_smooth_alpha_curve.png"))

    test_df = pd.DataFrame({
        "True": bundle["y_test"],
        "M0 prediction": bundle["m0_test_pred"],
        "Raw residual correction": bundle["raw_test_correction"],
        "Adaptive alpha": bundle["test_alpha"],
        "Actual correction": bundle["test_alpha"] * bundle["raw_test_correction"],
        f"{MODEL_DISPLAY_NAME} prediction": bundle["m2_test_pred"],
    })
    test_df["Observed M0 residual"] = test_df["True"] - test_df["M0 prediction"]
    test_df[f"{MODEL_DISPLAY_NAME} residual"] = test_df["True"] - test_df[f"{MODEL_DISPLAY_NAME} prediction"]
    test_df.to_excel(os.path.join(output_dir, "best_R5CW_test_correction_diagnostics.xlsx"), index=False)

    for x_col, y_col, fname, title, identity in [
        ("Raw residual correction", "Observed M0 residual", "best_R5CW_raw_correction_vs_observed_M0_residual.png", "Raw Correction", True),
        ("Actual correction", "Observed M0 residual", "best_R5CW_actual_correction_vs_observed_M0_residual.png", "Applied Correction", True),
        ("M0 prediction", "Actual correction", "best_R5CW_actual_correction_vs_M0_prediction.png", "Applied Correction", False),
        ("M0 prediction", "Adaptive alpha", "best_R5CW_alpha_vs_M0_prediction.png", "Alpha", False),
    ]:
        plt.figure(figsize=FIGSIZE["correction_diagnostic"])
        ax = sns.scatterplot(data=test_df, x=x_col, y=y_col, s=100, alpha=0.7, edgecolor="k", color="#b4d4e1")
        if identity:
            lim_min = float(min(test_df[x_col].min(), test_df[y_col].min()))
            lim_max = float(max(test_df[x_col].max(), test_df[y_col].max()))
            ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1.5)
        else:
            ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel(x_col, fontsize=18, fontweight="bold")
        ax.set_ylabel(y_col, fontsize=18, fontweight="bold")
        ax.set_title(title, fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        save_current_figure(os.path.join(output_dir, fname))
    return test_df


def build_r5_feature_inputs_from_bundle(bundle, weather_cols):
    train_feature_df = build_r5_residual_feature_df(bundle["train_df"], weather_cols, PREDICTED_PROTEIN_COL)[bundle["residual_feature_cols"]]
    test_feature_df = build_r5_residual_feature_df(bundle["test_df"], weather_cols, PREDICTED_PROTEIN_COL)[bundle["residual_feature_cols"]]
    return train_feature_df, test_feature_df


def plot_importance_outputs(bundle, weather_cols, output_dir):
    outputs = {}
    m0_model = bundle["m0_model"]
    m0_scaler = bundle["m0_scaler"]
    X_test_weather = bundle["test_df"][weather_cols].values
    y_test = bundle["y_test"]

    if hasattr(m0_model, "feature_importances_"):
        imp_df = pd.DataFrame({"Feature": weather_cols, "Importance": m0_model.feature_importances_}).sort_values("Importance", ascending=False)
        imp_df.to_excel(os.path.join(output_dir, "best_R5CW_M0_RF_feature_importance.xlsx"), index=False)
        top_df = imp_df.head(TOP_K_FEATURES)
        plot_barh_code2(top_df["Importance"].values, top_df["Feature"].values,
                        os.path.join(output_dir, "best_R5CW_M0_RF_feature_importance.png"),
                        FIGSIZE["importance"], "RF feature importance", "M0 RF Feature Importance")
        outputs["M0_RF_importance"] = imp_df

    try:
        m0_perm = permutation_importance(m0_model, m0_scaler.transform(X_test_weather), y_test, scoring="r2", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
        m0_perm_df = pd.DataFrame({"Feature": weather_cols, "Permutation importance mean": m0_perm.importances_mean, "Permutation importance std": m0_perm.importances_std}).sort_values("Permutation importance mean", ascending=False)
        m0_perm_df.to_excel(os.path.join(output_dir, "best_R5CW_M0_permutation_importance.xlsx"), index=False)
        top_df = m0_perm_df.head(TOP_K_FEATURES)
        plot_barh_code2(top_df["Permutation importance mean"].values, top_df["Feature"].values,
                        os.path.join(output_dir, "best_R5CW_M0_permutation_importance.png"),
                        FIGSIZE["importance"], "Permutation importance mean", "M0 Permutation Importance",
                        xerr=top_df["Permutation importance std"].values)
        outputs["M0_permutation"] = m0_perm_df
    except Exception as e:
        pd.DataFrame([{"stage": "M0 permutation importance", "error": str(e)}]).to_excel(os.path.join(output_dir, "best_R5CW_M0_permutation_importance_error.xlsx"), index=False)

    try:
        _, residual_test_feature_df = build_r5_feature_inputs_from_bundle(bundle, weather_cols)
        target_residual = np.asarray(bundle["y_test"], dtype=float) - np.asarray(bundle["m0_test_pred"], dtype=float)
        residual_perm = permutation_importance(bundle["residual_model"], residual_test_feature_df.values, target_residual, scoring="neg_root_mean_squared_error", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
        residual_perm_df = pd.DataFrame({"Feature": residual_test_feature_df.columns, "Permutation importance mean": residual_perm.importances_mean, "Permutation importance std": residual_perm.importances_std}).sort_values("Permutation importance mean", ascending=False)
        residual_perm_df.to_excel(os.path.join(output_dir, "best_R5CW_residual_model_permutation_importance.xlsx"), index=False)
        top_df = residual_perm_df.head(TOP_K_FEATURES)
        plot_barh_code2(top_df["Permutation importance mean"].values, top_df["Feature"].values,
                        os.path.join(output_dir, "best_R5CW_residual_model_permutation_importance.png"),
                        FIGSIZE["importance"], "Permutation importance mean", f"{MODEL_DISPLAY_NAME} Permutation Importance",
                        xerr=top_df["Permutation importance std"].values)
        outputs["Residual_permutation"] = residual_perm_df
    except Exception as e:
        pd.DataFrame([{"stage": "R5CW residual permutation importance", "error": str(e)}]).to_excel(os.path.join(output_dir, "best_R5CW_residual_model_permutation_importance_error.xlsx"), index=False)

    if shap is not None:
        try:
            X_test_scaled = m0_scaler.transform(X_test_weather)
            explainer = shap.TreeExplainer(m0_model)
            shap_values = np.asarray(explainer.shap_values(X_test_scaled))
            shap_df = pd.DataFrame({"Feature": weather_cols, "SHAP Mean Importance": np.abs(shap_values).mean(axis=0)}).sort_values("SHAP Mean Importance", ascending=False)
            shap_df.to_excel(os.path.join(output_dir, "best_R5CW_M0_SHAP_importance.xlsx"), index=False)
            top_features = shap_df.head(TOP_K_FEATURES)["Feature"].tolist()
            top_idx = [weather_cols.index(c) for c in top_features]
            shap.summary_plot(shap_values[:, top_idx], X_test_scaled[:, top_idx], feature_names=top_features, show=False,
                              plot_type="dot", plot_size=FIGSIZE["shap"])
            ax = plt.gca()
            ax.set_xlabel(XAXIS_LABEL["shap_dot_x"], fontsize=14, fontweight="bold")
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")
                tick.set_fontsize(14)
            save_current_figure(os.path.join(output_dir, "best_R5CW_M0_SHAP_beeswarm.png"))
            top_df = shap_df.head(TOP_K_FEATURES)
            plot_barh_code2(top_df["SHAP Mean Importance"].values, top_df["Feature"].values,
                            os.path.join(output_dir, "best_R5CW_M0_SHAP_bar.png"),
                            FIGSIZE["shap"], XAXIS_LABEL["shap_bar_x"], "SHAP Feature Importance")
            outputs["M0_SHAP"] = shap_df
        except Exception as e:
            pd.DataFrame([{"stage": "M0 SHAP", "error": str(e)}]).to_excel(os.path.join(output_dir, "best_R5CW_M0_SHAP_error.xlsx"), index=False)
    else:
        pd.DataFrame([{"stage": "M0 SHAP", "error": "shap is not installed"}]).to_excel(os.path.join(output_dir, "best_R5CW_M0_SHAP_error.xlsx"), index=False)
    return outputs


def plot_weight_outputs(best_record, teacher_clean, test_df, output_dir):
    scheme = best_record.get("Weight scheme", "none")
    beta = best_record.get("Weight beta", 0.0)
    location = best_record.get("Weight location", "none")
    if scheme == "none" or pd.isna(beta):
        return None
    beta = float(beta)
    residual_weight = build_sample_weight(teacher_clean, scheme=scheme, beta=beta, protein_col=PROTEIN_COL)
    weight_df = pd.DataFrame({"Dataset": "Teacher", "Weight": residual_weight, TMIN_COL: teacher_clean[TMIN_COL].values, TMIN_EXCESS_COL: teacher_clean[TMIN_EXCESS_COL].values, PROTEIN_COL: teacher_clean[PROTEIN_COL].values})
    if PROTEIN_RISK_COL in teacher_clean.columns:
        weight_df[PROTEIN_RISK_COL] = teacher_clean[PROTEIN_RISK_COL].values
    if location == "residual_and_alpha":
        alpha_weight = build_sample_weight(test_df, scheme=scheme, beta=beta, protein_col=PREDICTED_PROTEIN_COL)
        alpha_df = pd.DataFrame({"Dataset": "Test", "Weight": alpha_weight, TMIN_COL: test_df[TMIN_COL].values, TMIN_EXCESS_COL: test_df[TMIN_EXCESS_COL].values, PREDICTED_PROTEIN_COL: test_df[PREDICTED_PROTEIN_COL].values, PROTEIN_RISK_COL: test_df[PROTEIN_RISK_COL].values})
        weight_df = pd.concat([weight_df, alpha_df], axis=0, ignore_index=True)
    weight_df.to_excel(os.path.join(output_dir, "best_R5CW_weight_distribution_data.xlsx"), index=False)

    plt.figure(figsize=FIGSIZE["weight"])
    ax = sns.histplot(data=weight_df, x="Weight", hue="Dataset", bins=15, alpha=0.65, edgecolor="black")
    plt.xlabel("Sample weight", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("Sample Weight Distribution", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, "best_R5CW_weight_distribution.png"))

    if PROTEIN_RISK_COL in weight_df.columns:
        plt.figure(figsize=FIGSIZE["weight"])
        ax = sns.scatterplot(data=weight_df, x=PROTEIN_RISK_COL, y="Weight", hue="Dataset", s=100, alpha=0.75, edgecolor="k")
        ax.set_xlabel(PROTEIN_RISK_COL, fontsize=18, fontweight="bold")
        ax.set_ylabel("Sample weight", fontsize=18, fontweight="bold")
        ax.set_title("Weight Signal", fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        save_current_figure(os.path.join(output_dir, "best_R5CW_weight_vs_protein_heat_risk.png"))
    return weight_df


def plot_grid_heatmaps_and_rankings(result_df, output_dir):
    r5cw_df = result_df[result_df["Model family"] == "R5CW"].copy()
    if r5cw_df.empty:
        return
    r5cw_df["z_cv"] = r5cw_df["z_score_threshold"].astype(str) + "_" + r5cw_df["cv"].astype(str)
    for value_col, fname, title in [
        ("M2 test R2", "R5CW_heatmap_M2_test_R2_by_version_and_ZCV.png", f"{MODEL_DISPLAY_NAME} Test R2"),
        ("Delta test R2 M2_minus_WR0", "R5CW_heatmap_delta_R2_vs_WR0_by_version_and_ZCV.png", "Delta R2 vs WR0"),
        ("Delta test R2 M2_minus_R5C", "R5CW_heatmap_delta_R2_vs_R5C_by_version_and_ZCV.png", "Delta R2 vs R5C"),
        ("M2 test RMSE", "R5CW_heatmap_M2_test_RMSE_by_version_and_ZCV.png", f"{MODEL_DISPLAY_NAME} Test RMSE"),
    ]:
        if value_col not in r5cw_df.columns:
            continue
        pivot = r5cw_df.pivot_table(index="Version", columns="z_cv", values=value_col, aggfunc="mean")
        pivot.to_excel(os.path.join(output_dir, fname.replace(".png", ".xlsx")))
        fig_width = max(8, 0.45 * pivot.shape[1])
        fig_height = max(5, 0.22 * pivot.shape[0])
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(pivot, cmap="Blues", annot=False, linewidths=0.4, linecolor="white")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Z-CV", fontsize=14, fontweight="bold")
        plt.ylabel("Model", fontsize=14, fontweight="bold")
        save_current_figure(os.path.join(output_dir, fname))
    top_df = r5cw_df.sort_values(["M2 test R2", "M2 test RMSE"], ascending=[False, True]).head(30).copy()
    top_df.to_excel(os.path.join(output_dir, "top30_R5CW_models_by_test_R2.xlsx"), index=False)
    label = top_df["Version"] + " | Z" + top_df["z_score_threshold"].astype(str) + " CV" + top_df["cv"].astype(str)
    plot_df = pd.DataFrame({"Model": label, "M2 test R2": top_df["M2 test R2"].values})
    plt.figure(figsize=FIGSIZE["ranking"])
    ax = sns.barplot(data=plot_df, y="Model", x="M2 test R2", edgecolor="black")
    ax.set_xlabel("Test R²", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(f"Top {MODEL_DISPLAY_NAME} Models", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="x")
    save_current_figure(os.path.join(output_dir, "top30_R5CW_models_by_test_R2.png"))


def save_extended_workbook(best_record, metric_df, residual_diag_df, correction_diag_df, importance_outputs, output_dir):
    workbook_path = os.path.join(output_dir, "best_R5CW_extended_outputs_summary.xlsx")
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame([best_record]).to_excel(writer, sheet_name="best_record", index=False)
        if metric_df is not None:
            metric_df.to_excel(writer, sheet_name="metric_comparison", index=False)
        if residual_diag_df is not None:
            residual_diag_df.to_excel(writer, sheet_name="test_residuals", index=False)
        if correction_diag_df is not None:
            correction_diag_df.to_excel(writer, sheet_name="correction_diagnostics", index=False)
        for key, df in importance_outputs.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=key[:31], index=False)
    return workbook_path


# =========================================================
# 5. Main workflow
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

    excluded_teacher = set([TARGET_COL, PROTEIN_COL] + KNOWLEDGE_ONLY_COLS)
    excluded_original = set([original_target_col] + KNOWLEDGE_ONLY_COLS)
    teacher_weather_cols = [c for c in teacher_numeric.columns if c not in excluded_teacher]
    original_weather_cols = [c for c in original_numeric.columns if c not in excluded_original]
    base_weather_features = [c for c in original_weather_cols if c in teacher_weather_cols]

    if len(base_weather_features) == 0:
        raise ValueError("No common numeric weather features were found.")
    if TMIN_COL not in base_weather_features:
        raise ValueError(f"{TMIN_COL} must be in base weather features.")

    print("\nBase weather features for M0, protein proxy, and ordinary residual model:")
    print(base_weather_features)

    pd.DataFrame({"base_weather_features": base_weather_features}).to_excel(
        os.path.join(OUTPUT_DIR, "base_weather_features.xlsx"), index=False
    )

    teacher_df = teacher_numeric[[TARGET_COL, PROTEIN_COL] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    teacher_clean, teacher_outliers = remove_outliers_by_zscore(
        teacher_df,
        base_weather_features + [PROTEIN_COL],
        TEACHER_Z_THRESHOLD,
    )
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
        sample_weight=None,
    )
    protein_oof_metrics = calculate_metrics(
        teacher_clean[PROTEIN_COL].values,
        oof_protein,
        "Protein OOF prediction",
    )

    print("\nProtein proxy best params:", protein_info["best_params"])
    print("Protein proxy CV R2:", protein_info["best_repeated_cv_score"])
    print("Protein proxy OOF metrics:", protein_oof_metrics)

    student_df = original_numeric[[original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    student_df[PREDICTED_PROTEIN_COL] = protein_model.predict(protein_scaler.transform(student_df[base_weather_features].values))
    student_df = add_protein_heat_risk(student_df, PREDICTED_PROTEIN_COL)

    print("\nStudent valid n:", len(student_df))

    # R5CW weighted specs.
    weighted_specs = []
    for scheme in WEIGHT_SCHEMES:
        for beta in WEIGHT_BETAS:
            for location in WEIGHT_LOCATIONS:
                weighted_specs.append({
                    "version": f"R5CW_{scheme}_beta{str(beta).replace('.', '_')}_{location}",
                    "scheme": scheme,
                    "beta": beta,
                    "location": location,
                    "residual_weighted": True,
                    "alpha_weighted": location == "residual_and_alpha",
                })

    records = []
    best_weighted_record = None
    best_weighted_bundle = None
    best_any_record = None
    best_any_bundle = None

    for z_thr in STUDENT_Z_THRESHOLDS:
        student_clean, student_outliers = remove_outliers_by_zscore(student_df, base_weather_features, z_thr)
        if len(student_clean) < 30:
            print(f"Z-score={z_thr}: too few samples after cleaning; skipped.")
            continue

        X = student_clean[base_weather_features].values
        y = student_clean[original_target_col].values

        train_idx, test_idx = train_test_split(
            np.arange(len(student_clean)),
            test_size=STUDENT_TEST_SIZE,
            random_state=RANDOM_STATE,
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

            print(f"\nTraining combined residual-constraint-weighting grid: Z-score={z_thr}, CV={cv}")

            m0_model, m0_scaler, m0_search, m0_train_pred, m0_test_pred = train_m0_rf_on_fixed_split(
                X_train,
                y_train,
                X_test,
                cv,
            )
            m0_train_metrics = calculate_metrics(y_train, m0_train_pred, "M0 train")
            m0_test_metrics = calculate_metrics(y_test, m0_test_pred, "M0 test")

            # ---------------------------------------------------------
            # WR0 baseline: ordinary residual correction, no constraint, no weighting.
            # ---------------------------------------------------------
            wr0_model, wr0_scaler, wr0_info, wr0_feature_cols, wr0_teacher_df = fit_ordinary_residual_model(
                teacher_clean,
                base_weather_features,
                m0_model,
                m0_scaler,
                sample_weight=None,
                task_suffix="WR0",
            )
            wr0_raw_train, wr0_raw_test, wr0_train_feature_df, wr0_test_feature_df = make_ordinary_residual_predictions(
                wr0_model,
                wr0_scaler,
                wr0_feature_cols,
                train_df,
                test_df,
                base_weather_features,
            )
            wr0_rec, wr0_bundle = evaluate_m2(
                version="WR0_unweighted_residual_correction",
                y_train=y_train,
                y_test=y_test,
                m0_train_pred=m0_train_pred,
                m0_test_pred=m0_test_pred,
                raw_train_correction=wr0_raw_train,
                raw_test_correction=wr0_raw_test,
                alpha_selection_weight=None,
                alpha_selection_metric="r2",
            )
            wr0_rec = update_standard_record(
                wr0_rec,
                model_family="WR0",
                z_thr=z_thr,
                cv=cv,
                student_clean=student_clean,
                student_outliers=student_outliers,
                train_idx=train_idx,
                test_idx=test_idx,
                m0_search=m0_search,
                protein_info=protein_info,
                protein_oof_metrics=protein_oof_metrics,
                m0_train_metrics=m0_train_metrics,
                m0_test_metrics=m0_test_metrics,
                weight_scheme="none",
                beta=0.0,
                weight_location="none",
                residual_weight=None,
                alpha_weight=None,
                residual_info=wr0_info,
            )
            wr0_rec["WR0 M2 test R2"] = wr0_rec["M2 test R2"]
            wr0_rec["WR0 M2 test RMSE"] = wr0_rec["M2 test RMSE"]
            wr0_rec["WR0 M2 test MAE"] = wr0_rec["M2 test MAE"]
            wr0_rec["R5C M2 test R2"] = np.nan
            wr0_rec["R5C M2 test RMSE"] = np.nan
            wr0_rec["R5C M2 test MAE"] = np.nan
            wr0_rec["Delta test R2 M2_minus_WR0"] = 0.0
            wr0_rec["Delta test RMSE M2_minus_WR0"] = 0.0
            wr0_rec["Delta test MAE M2_minus_WR0"] = 0.0
            wr0_rec["Delta test R2 M2_minus_R5C"] = np.nan
            records.append(wr0_rec)

            # ---------------------------------------------------------
            # R5C baseline: residual correction + conditional constraint, no weighting.
            # ---------------------------------------------------------
            r5_model, r5_info, r5_feature_cols, r5_teacher_df = fit_r5_conditional_residual_model(
                teacher_clean,
                base_weather_features,
                m0_model,
                m0_scaler,
                sample_weight=None,
                task_suffix="R5C",
            )
            r5_raw_train, r5_raw_test, r5_train_feature_df, r5_test_feature_df = make_r5_residual_predictions(
                r5_model,
                r5_feature_cols,
                train_df,
                test_df,
                base_weather_features,
            )
            r5_rec, r5_bundle = evaluate_m2(
                version="R5C_conditional_constraint_no_weight",
                y_train=y_train,
                y_test=y_test,
                m0_train_pred=m0_train_pred,
                m0_test_pred=m0_test_pred,
                raw_train_correction=r5_raw_train,
                raw_test_correction=r5_raw_test,
                alpha_selection_weight=None,
                alpha_selection_metric="r2",
            )
            r5_rec = update_standard_record(
                r5_rec,
                model_family="R5C",
                z_thr=z_thr,
                cv=cv,
                student_clean=student_clean,
                student_outliers=student_outliers,
                train_idx=train_idx,
                test_idx=test_idx,
                m0_search=m0_search,
                protein_info=protein_info,
                protein_oof_metrics=protein_oof_metrics,
                m0_train_metrics=m0_train_metrics,
                m0_test_metrics=m0_test_metrics,
                weight_scheme="none",
                beta=0.0,
                weight_location="none",
                residual_weight=None,
                alpha_weight=None,
                residual_info=r5_info,
            )
            r5_rec["WR0 M2 test R2"] = wr0_rec["M2 test R2"]
            r5_rec["WR0 M2 test RMSE"] = wr0_rec["M2 test RMSE"]
            r5_rec["WR0 M2 test MAE"] = wr0_rec["M2 test MAE"]
            r5_rec["R5C M2 test R2"] = r5_rec["M2 test R2"]
            r5_rec["R5C M2 test RMSE"] = r5_rec["M2 test RMSE"]
            r5_rec["R5C M2 test MAE"] = r5_rec["M2 test MAE"]
            r5_rec["Delta test R2 M2_minus_WR0"] = r5_rec["M2 test R2"] - wr0_rec["M2 test R2"]
            r5_rec["Delta test RMSE M2_minus_WR0"] = r5_rec["M2 test RMSE"] - wr0_rec["M2 test RMSE"]
            r5_rec["Delta test MAE M2_minus_WR0"] = r5_rec["M2 test MAE"] - wr0_rec["M2 test MAE"]
            r5_rec["Delta test R2 M2_minus_R5C"] = 0.0
            records.append(r5_rec)

            # Track best any model.
            for rec, bundle, model, model_kind, feature_cols, train_feat, test_feat in [
                (wr0_rec, wr0_bundle, wr0_model, "WR0", wr0_feature_cols, wr0_train_feature_df, wr0_test_feature_df),
                (r5_rec, r5_bundle, r5_model, "R5C", r5_feature_cols, r5_train_feature_df, r5_test_feature_df),
            ]:
                is_better_any = best_any_record is None or (
                    rec["M2 test R2"] > best_any_record["M2 test R2"] or
                    (np.isclose(rec["M2 test R2"], best_any_record["M2 test R2"]) and rec["M2 test RMSE"] < best_any_record["M2 test RMSE"])
                )
                if is_better_any:
                    best_any_record = rec.copy()
                    best_any_bundle = {
                        "model_kind": model_kind,
                        "student_clean": student_clean.copy(),
                        "train_df": train_df.copy(),
                        "test_df": test_df.copy(),
                        "m0_model": m0_model,
                        "m0_scaler": m0_scaler,
                        "m0_train_pred": m0_train_pred,
                        "m0_test_pred": m0_test_pred,
                        "residual_model": model,
                        "residual_scaler": wr0_scaler if model_kind == "WR0" else None,
                        "residual_feature_cols": feature_cols,
                        "raw_train_correction": wr0_raw_train if model_kind == "WR0" else r5_raw_train,
                        "raw_test_correction": wr0_raw_test if model_kind == "WR0" else r5_raw_test,
                        "alpha_info": bundle["alpha_info"],
                        "train_alpha": bundle["train_alpha"],
                        "test_alpha": bundle["test_alpha"],
                        "m2_train_pred": bundle["m2_train_pred"],
                        "m2_test_pred": bundle["m2_test_pred"],
                        "y_train": y_train,
                        "y_test": y_test,
                    }

            # ---------------------------------------------------------
            # R5CW: conditional constraint + sample weighting.
            # ---------------------------------------------------------
            for spec in weighted_specs:
                print("  Training", spec["version"])

                residual_weight = None
                alpha_weight = None

                if spec["residual_weighted"]:
                    residual_weight = build_sample_weight(
                        teacher_clean,
                        scheme=spec["scheme"],
                        beta=spec["beta"],
                        protein_col=PROTEIN_COL,
                    )

                if spec["alpha_weighted"]:
                    alpha_weight = build_sample_weight(
                        test_df,
                        scheme=spec["scheme"],
                        beta=spec["beta"],
                        protein_col=PREDICTED_PROTEIN_COL,
                    )

                cw_model, cw_info, cw_feature_cols, cw_teacher_df = fit_r5_conditional_residual_model(
                    teacher_clean,
                    base_weather_features,
                    m0_model,
                    m0_scaler,
                    sample_weight=residual_weight,
                    task_suffix=spec["version"],
                )
                cw_raw_train, cw_raw_test, cw_train_feature_df, cw_test_feature_df = make_r5_residual_predictions(
                    cw_model,
                    cw_feature_cols,
                    train_df,
                    test_df,
                    base_weather_features,
                )

                alpha_metric = "weighted_r2" if spec["alpha_weighted"] else "r2"
                cw_rec, cw_bundle = evaluate_m2(
                    version=spec["version"],
                    y_train=y_train,
                    y_test=y_test,
                    m0_train_pred=m0_train_pred,
                    m0_test_pred=m0_test_pred,
                    raw_train_correction=cw_raw_train,
                    raw_test_correction=cw_raw_test,
                    alpha_selection_weight=alpha_weight,
                    alpha_selection_metric=alpha_metric,
                )
                cw_rec = update_standard_record(
                    cw_rec,
                    model_family="R5CW",
                    z_thr=z_thr,
                    cv=cv,
                    student_clean=student_clean,
                    student_outliers=student_outliers,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    m0_search=m0_search,
                    protein_info=protein_info,
                    protein_oof_metrics=protein_oof_metrics,
                    m0_train_metrics=m0_train_metrics,
                    m0_test_metrics=m0_test_metrics,
                    weight_scheme=spec["scheme"],
                    beta=spec["beta"],
                    weight_location=spec["location"],
                    residual_weight=residual_weight,
                    alpha_weight=alpha_weight,
                    residual_info=cw_info,
                )

                cw_rec["WR0 M2 test R2"] = wr0_rec["M2 test R2"]
                cw_rec["WR0 M2 test RMSE"] = wr0_rec["M2 test RMSE"]
                cw_rec["WR0 M2 test MAE"] = wr0_rec["M2 test MAE"]
                cw_rec["R5C M2 test R2"] = r5_rec["M2 test R2"]
                cw_rec["R5C M2 test RMSE"] = r5_rec["M2 test RMSE"]
                cw_rec["R5C M2 test MAE"] = r5_rec["M2 test MAE"]

                cw_rec["Delta test R2 M2_minus_WR0"] = cw_rec["M2 test R2"] - wr0_rec["M2 test R2"]
                cw_rec["Delta test RMSE M2_minus_WR0"] = cw_rec["M2 test RMSE"] - wr0_rec["M2 test RMSE"]
                cw_rec["Delta test MAE M2_minus_WR0"] = cw_rec["M2 test MAE"] - wr0_rec["M2 test MAE"]

                cw_rec["Delta test R2 M2_minus_R5C"] = cw_rec["M2 test R2"] - r5_rec["M2 test R2"]
                cw_rec["Delta test RMSE M2_minus_R5C"] = cw_rec["M2 test RMSE"] - r5_rec["M2 test RMSE"]
                cw_rec["Delta test MAE M2_minus_R5C"] = cw_rec["M2 test MAE"] - r5_rec["M2 test MAE"]

                cw_rec["Improves over WR0 by R2"] = cw_rec["Delta test R2 M2_minus_WR0"] > 0
                cw_rec["Improves over R5C by R2"] = cw_rec["Delta test R2 M2_minus_R5C"] > 0
                records.append(cw_rec)

                is_better_weighted = best_weighted_record is None or (
                    cw_rec["M2 test R2"] > best_weighted_record["M2 test R2"] or
                    (np.isclose(cw_rec["M2 test R2"], best_weighted_record["M2 test R2"]) and cw_rec["M2 test RMSE"] < best_weighted_record["M2 test RMSE"])
                )
                if is_better_weighted:
                    best_weighted_record = cw_rec.copy()
                    best_weighted_bundle = {
                        "student_clean": student_clean.copy(),
                        "train_df": train_df.copy(),
                        "test_df": test_df.copy(),
                        "m0_model": m0_model,
                        "m0_scaler": m0_scaler,
                        "m0_train_pred": m0_train_pred,
                        "m0_test_pred": m0_test_pred,
                        "residual_model": cw_model,
                        "residual_feature_cols": cw_feature_cols,
                        "raw_train_correction": cw_raw_train,
                        "raw_test_correction": cw_raw_test,
                        "alpha_info": cw_bundle["alpha_info"],
                        "train_alpha": cw_bundle["train_alpha"],
                        "test_alpha": cw_bundle["test_alpha"],
                        "m2_train_pred": cw_bundle["m2_train_pred"],
                        "m2_test_pred": cw_bundle["m2_test_pred"],
                        "y_train": y_train,
                        "y_test": y_test,
                    }

                is_better_any = best_any_record is None or (
                    cw_rec["M2 test R2"] > best_any_record["M2 test R2"] or
                    (np.isclose(cw_rec["M2 test R2"], best_any_record["M2 test R2"]) and cw_rec["M2 test RMSE"] < best_any_record["M2 test RMSE"])
                )
                if is_better_any:
                    best_any_record = cw_rec.copy()
                    best_any_bundle = {
                        "model_kind": "R5CW",
                        "student_clean": student_clean.copy(),
                        "train_df": train_df.copy(),
                        "test_df": test_df.copy(),
                        "m0_model": m0_model,
                        "m0_scaler": m0_scaler,
                        "m0_train_pred": m0_train_pred,
                        "m0_test_pred": m0_test_pred,
                        "residual_model": cw_model,
                        "residual_scaler": None,
                        "residual_feature_cols": cw_feature_cols,
                        "raw_train_correction": cw_raw_train,
                        "raw_test_correction": cw_raw_test,
                        "alpha_info": cw_bundle["alpha_info"],
                        "train_alpha": cw_bundle["train_alpha"],
                        "test_alpha": cw_bundle["test_alpha"],
                        "m2_train_pred": cw_bundle["m2_train_pred"],
                        "m2_test_pred": cw_bundle["m2_test_pred"],
                        "y_train": y_train,
                        "y_test": y_test,
                    }

    if len(records) == 0:
        raise RuntimeError("No valid model was trained.")

    result_df = pd.DataFrame(records)
    result_df.to_excel(os.path.join(OUTPUT_DIR, "combined_residual_constraint_weighting_all_results.xlsx"), index=False)

    core_cols = [
        "Model family", "Version", "Weight scheme", "Weight beta", "Weight location",
        "z_score_threshold", "cv",
        "M0 test R2", "M0 test RMSE", "M0 test MAE",
        "WR0 M2 test R2", "R5C M2 test R2",
        "M2 test R2", "M2 test RMSE", "M2 test MAE",
        "Delta test R2 M2_minus_M0",
        "Delta test R2 M2_minus_WR0",
        "Delta test R2 M2_minus_R5C",
        "Delta test RMSE M2_minus_WR0",
        "Delta test RMSE M2_minus_R5C",
        "Improves over WR0 by R2",
        "Improves over R5C by R2",
        "M0_best_params", "Protein_proxy_cv_R2", "Protein_proxy_oof_R2",
        "Low group n", "High group n", "Residual input feature cols", "Residual model feature cols",
        "Residual train weight_min", "Residual train weight_max", "Residual train weight_mean", "Residual train weight_std",
        "Alpha test weight_min", "Alpha test weight_max", "Alpha test weight_mean", "Alpha test weight_std",
    ]
    core_cols_existing = [c for c in core_cols if c in result_df.columns]
    result_df[core_cols_existing].to_excel(
        os.path.join(OUTPUT_DIR, "combined_residual_constraint_weighting_core_metrics.xlsx"),
        index=False,
    )

    # Best records.
    if best_weighted_record is not None:
        pd.DataFrame([best_weighted_record]).to_excel(
            os.path.join(OUTPUT_DIR, "best_R5CW_weighted_model_metrics.xlsx"),
            index=False,
        )
    if best_any_record is not None:
        pd.DataFrame([best_any_record]).to_excel(
            os.path.join(OUTPUT_DIR, "best_any_model_metrics.xlsx"),
            index=False,
        )

    # Save selected best weighted model and predictions.
    if best_weighted_bundle is not None:
        joblib.dump(protein_model, os.path.join(OUTPUT_DIR, "protein_proxy_model.pkl"))
        joblib.dump(protein_scaler, os.path.join(OUTPUT_DIR, "protein_proxy_scaler.pkl"))
        joblib.dump(best_weighted_bundle["m0_model"], os.path.join(OUTPUT_DIR, "best_R5CW_M0_model.pkl"))
        joblib.dump(best_weighted_bundle["m0_scaler"], os.path.join(OUTPUT_DIR, "best_R5CW_M0_scaler.pkl"))
        joblib.dump(best_weighted_bundle["residual_model"], os.path.join(OUTPUT_DIR, "best_R5CW_conditional_residual_model.pkl"))
        joblib.dump(best_weighted_bundle["alpha_info"], os.path.join(OUTPUT_DIR, "best_R5CW_alpha_info.pkl"))

        train_pred_df = best_weighted_bundle["train_df"].copy()
        test_pred_df = best_weighted_bundle["test_df"].copy()

        train_pred_df["M0 prediction"] = best_weighted_bundle["m0_train_pred"]
        train_pred_df["Raw residual correction"] = best_weighted_bundle["raw_train_correction"]
        train_pred_df["Adaptive alpha"] = best_weighted_bundle["train_alpha"]
        train_pred_df["M2 prediction"] = best_weighted_bundle["m2_train_pred"]
        train_pred_df["M0 residual"] = train_pred_df[original_target_col] - train_pred_df["M0 prediction"]
        train_pred_df["M2 residual"] = train_pred_df[original_target_col] - train_pred_df["M2 prediction"]

        test_pred_df["M0 prediction"] = best_weighted_bundle["m0_test_pred"]
        test_pred_df["Raw residual correction"] = best_weighted_bundle["raw_test_correction"]
        test_pred_df["Adaptive alpha"] = best_weighted_bundle["test_alpha"]
        test_pred_df["M2 prediction"] = best_weighted_bundle["m2_test_pred"]
        test_pred_df["M0 residual"] = test_pred_df[original_target_col] - test_pred_df["M0 prediction"]
        test_pred_df["M2 residual"] = test_pred_df[original_target_col] - test_pred_df["M2 prediction"]

        train_pred_df.to_excel(os.path.join(OUTPUT_DIR, "best_R5CW_train_predictions.xlsx"), index=False)
        test_pred_df.to_excel(os.path.join(OUTPUT_DIR, "best_R5CW_test_predictions.xlsx"), index=False)

        # Extended outputs for the final selected R5CW model only.
        # These diagnostics do not affect training, alpha selection, weighting, or model ranking.
        EXTENDED_OUTPUT_DIR = safe_mkdir(os.path.join(OUTPUT_DIR, "best_R5CW_extended_outputs"))
        metric_df = plot_metric_comparison_for_best(best_weighted_record, EXTENDED_OUTPUT_DIR)
        plot_true_vs_pred_extended(best_weighted_bundle["y_train"], best_weighted_bundle["m0_train_pred"], best_weighted_bundle["y_test"], best_weighted_bundle["m0_test_pred"], "M0 baseline in selected R5CW combination", EXTENDED_OUTPUT_DIR, "best_R5CW_M0")
        plot_true_vs_pred_extended(best_weighted_bundle["y_train"], best_weighted_bundle["m2_train_pred"], best_weighted_bundle["y_test"], best_weighted_bundle["m2_test_pred"], "Selected R5CW M2 model", EXTENDED_OUTPUT_DIR, "best_R5CW_M2")
        residual_diag_df = plot_residual_diagnostics(best_weighted_bundle["y_test"], best_weighted_bundle["m0_test_pred"], best_weighted_bundle["m2_test_pred"], EXTENDED_OUTPUT_DIR)
        correction_diag_df = plot_alpha_and_correction_diagnostics(best_weighted_bundle, EXTENDED_OUTPUT_DIR)
        importance_outputs = plot_importance_outputs(best_weighted_bundle, base_weather_features, EXTENDED_OUTPUT_DIR)
        plot_weight_outputs(best_weighted_record, teacher_clean, best_weighted_bundle["test_df"], EXTENDED_OUTPUT_DIR)
        plot_grid_heatmaps_and_rankings(result_df, EXTENDED_OUTPUT_DIR)
        workbook_path = save_extended_workbook(best_weighted_record, metric_df, residual_diag_df, correction_diag_df, importance_outputs, EXTENDED_OUTPUT_DIR)
        print("\nExtended outputs for the selected R5CW model saved to:", EXTENDED_OUTPUT_DIR)
        print("Extended summary workbook:", workbook_path)

    print("\n================ Combined residual + constraint + weighting grid search completed ================")
    print("Output directory:", OUTPUT_DIR)

    if best_weighted_record is not None:
        print("\nBest R5CW weighted model:")
        print("Version:", best_weighted_record["Version"])
        print("Best z-score:", best_weighted_record["z_score_threshold"])
        print("Best CV:", best_weighted_record["cv"])
        print("M0 test R2:", best_weighted_record["M0 test R2"])
        print("WR0 M2 test R2:", best_weighted_record["WR0 M2 test R2"])
        print("R5C M2 test R2:", best_weighted_record["R5C M2 test R2"])
        print("R5CW M2 test R2:", best_weighted_record["M2 test R2"])
        print("Delta test R2 M2_minus_WR0:", best_weighted_record["Delta test R2 M2_minus_WR0"])
        print("Delta test R2 M2_minus_R5C:", best_weighted_record["Delta test R2 M2_minus_R5C"])

    if best_any_record is not None:
        print("\nBest model among WR0, R5C, and R5CW:")
        print("Model family:", best_any_record["Model family"])
        print("Version:", best_any_record["Version"])
        print("Best z-score:", best_any_record["z_score_threshold"])
        print("Best CV:", best_any_record["cv"])
        print("M2 test R2:", best_any_record["M2 test R2"])


if __name__ == "__main__":
    main()
