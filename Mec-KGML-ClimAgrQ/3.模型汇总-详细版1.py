# -*- coding: utf-8 -*-
"""
M2 storage-protein knowledge-guided model:
Residual correction + conditional constraint learning + sample weighting.

Purpose
1. Build M0 baseline using only original common weather factors.
2. Build WR0 ordinary protein-guided residual correction:
      M2 = M0 prediction + alpha(M0 prediction) × residual correction
3. Build R5-only M0-based conditional residual constraint model:
      Low TMIN group: RF residual correction model without monotonic constraint
      High TMIN group: HGB residual correction model with monotonic constraint on Protein_Heat_Risk
4. Build R5 + sample-weighting models:
      Residual model training can be weighted by high-TMIN or protein-heat-risk knowledge.
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
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\数据库籼稻建模 - 汇总气象因子.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\3.模型汇总\模型汇总"
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
MODEL_DISPLAY_NAME = "M3"
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
ALPHA_SLOPE_CANDIDATES = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00]
ALPHA_CENTER_QUANTILE_CANDIDATES = [0.35, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
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

    M3 = M0 prediction + alpha(M0 prediction) * constrained correction

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
        M3 = M0 prediction + alpha(M0 prediction) * correction
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
        "M3 formulation": "M3 = M0 prediction + alpha(M0 prediction) * constrained correction",
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
        pd.DataFrame([{"stage": "M3 residual permutation importance", "error": str(e)}]).to_excel(os.path.join(output_dir, "best_R5CW_residual_model_permutation_importance_error.xlsx"), index=False)

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


def plot_model_screening_diagnostics(result_df, output_dir):
    """
    Plot model-screening and learning-effect diagnostics across all searched versions.
    These figures only summarize saved results and do not change model selection.
    """
    outputs = {}
    if result_df is None or result_df.empty:
        return outputs

    df = result_df.copy()
    screen_dir = safe_mkdir(os.path.join(output_dir, "learning_process_figures"))

    # 1. Model-family performance distribution.
    family_summary = (
        df.groupby("Model family", dropna=False)
        .agg(
            n_models=("Version", "count"),
            max_test_R2=("M2 test R2", "max"),
            mean_test_R2=("M2 test R2", "mean"),
            min_test_RMSE=("M2 test RMSE", "min"),
            mean_test_RMSE=("M2 test RMSE", "mean"),
            max_delta_R2_vs_M0=("Delta test R2 M2_minus_M0", "max"),
            mean_delta_R2_vs_M0=("Delta test R2 M2_minus_M0", "mean"),
        )
        .reset_index()
    )
    family_summary.to_excel(os.path.join(screen_dir, "model_family_screening_summary.xlsx"), index=False)
    outputs["model_family_summary"] = family_summary

    plt.figure(figsize=FIGSIZE["performance_bar"])
    ax = sns.boxplot(data=df, x="Model family", y="M2 test R2")
    sns.stripplot(data=df, x="Model family", y="M2 test R2", color="black", alpha=0.45, size=5)
    ax.set_xlabel("")
    ax.set_ylabel("Test R²", fontsize=14, fontweight="bold")
    ax.set_title("Model Screening", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(screen_dir, "model_family_test_R2_distribution.png"))

    plt.figure(figsize=FIGSIZE["performance_bar"])
    ax = sns.scatterplot(data=df, x="M2 test RMSE", y="M2 test R2", hue="Model family", s=90, edgecolor="black", alpha=0.80)
    ax.set_xlabel("Test RMSE", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test R²", fontsize=14, fontweight="bold")
    ax.set_title("Accuracy Trade-off", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(screen_dir, "model_family_R2_RMSE_tradeoff.png"))

    # 2. Improvement over M0 and over baselines.
    delta_cols = [
        ("Delta test R2 M2_minus_M0", "Delta R² vs M0", "delta_R2_vs_M0_by_family.png"),
        ("Delta test R2 M2_minus_WR0", "Delta R² vs WR0", "delta_R2_vs_WR0_by_family.png"),
        ("Delta test R2 M2_minus_R5C", "Delta R² vs R5C", "delta_R2_vs_R5C_by_family.png"),
    ]
    for col, ylabel, fname in delta_cols:
        if col not in df.columns:
            continue
        plot_df = df.dropna(subset=[col]).copy()
        if plot_df.empty:
            continue
        plt.figure(figsize=FIGSIZE["performance_bar"])
        ax = sns.boxplot(data=plot_df, x="Model family", y=col)
        sns.stripplot(data=plot_df, x="Model family", y=col, color="black", alpha=0.45, size=5)
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(ylabel, fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        save_current_figure(os.path.join(screen_dir, fname))

    # 3. Weight strategy heatmaps for R5CW.
    r5cw_df = df[df["Model family"] == "R5CW"].copy()
    if not r5cw_df.empty:
        for location in sorted(r5cw_df["Weight location"].dropna().unique()):
            sub = r5cw_df[r5cw_df["Weight location"] == location].copy()
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="Weight scheme", columns="Weight beta", values="M2 test R2", aggfunc="mean")
            pivot.to_excel(os.path.join(screen_dir, f"weight_strategy_test_R2_{location}.xlsx"))
            plt.figure(figsize=FIGSIZE["heatmap"])
            ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5, linecolor="white")
            ax.set_xlabel("Weight beta", fontsize=14, fontweight="bold")
            ax.set_ylabel("Weight scheme", fontsize=14, fontweight="bold")
            ax.set_title(f"Weight Search: {location}", fontsize=16, fontweight="bold")
            save_current_figure(os.path.join(screen_dir, f"weight_strategy_test_R2_{location}.png"))

        top_r5cw = r5cw_df.sort_values(["M2 test R2", "M2 test RMSE"], ascending=[False, True]).head(15).copy()
        top_r5cw["Label"] = (
            top_r5cw["Weight scheme"].astype(str)
            + " | beta="
            + top_r5cw["Weight beta"].astype(str)
            + " | "
            + top_r5cw["Weight location"].astype(str)
        )
        top_r5cw.to_excel(os.path.join(screen_dir, "top15_weighted_models.xlsx"), index=False)
        plt.figure(figsize=(8, max(4, 0.36 * len(top_r5cw))))
        ax = sns.barplot(data=top_r5cw, y="Label", x="M2 test R2", edgecolor="black")
        ax.set_xlabel("Test R²", fontsize=14, fontweight="bold")
        ax.set_ylabel("")
        ax.set_title("Top Weighted Models", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="x")
        save_current_figure(os.path.join(screen_dir, "top15_weighted_models_test_R2.png"))

    return outputs


def plot_alpha_search_diagnostics(best_bundle, output_dir):
    """
    Plot alpha-search diagnostics for the selected model.
    Uses the stored alpha search table from the selected model only.
    """
    outputs = {}
    alpha_df = best_bundle.get("alpha_df", None) if isinstance(best_bundle, dict) else None
    if alpha_df is None or not isinstance(alpha_df, pd.DataFrame) or alpha_df.empty:
        return outputs

    alpha_dir = safe_mkdir(os.path.join(output_dir, "learning_process_figures"))
    alpha_df = alpha_df.copy()
    alpha_df.to_excel(os.path.join(alpha_dir, "selected_model_alpha_search_all_candidates.xlsx"), index=False)
    outputs["alpha_search"] = alpha_df

    # Heatmap of alpha_base by alpha_amp.
    pivot_base_amp = alpha_df.pivot_table(
        index="alpha_base",
        columns="alpha_amp",
        values="selection_score",
        aggfunc="max",
    )
    pivot_base_amp.to_excel(os.path.join(alpha_dir, "alpha_search_base_amp_heatmap_data.xlsx"))
    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(pivot_base_amp, cmap="Blues", linewidths=0.4, linecolor="white")
    ax.set_xlabel("Alpha amp", fontsize=14, fontweight="bold")
    ax.set_ylabel("Alpha base", fontsize=14, fontweight="bold")
    ax.set_title("Alpha Search", fontsize=16, fontweight="bold")
    save_current_figure(os.path.join(alpha_dir, "alpha_search_base_amp_heatmap.png"))

    # Heatmap of slope by center quantile.
    pivot_slope_center = alpha_df.pivot_table(
        index="alpha_slope",
        columns="alpha_center_quantile",
        values="selection_score",
        aggfunc="max",
    )
    pivot_slope_center.to_excel(os.path.join(alpha_dir, "alpha_search_slope_center_heatmap_data.xlsx"))
    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(pivot_slope_center, cmap="Blues", linewidths=0.4, linecolor="white")
    ax.set_xlabel("Center quantile", fontsize=14, fontweight="bold")
    ax.set_ylabel("Alpha slope", fontsize=14, fontweight="bold")
    ax.set_title("Alpha Search", fontsize=16, fontweight="bold")
    save_current_figure(os.path.join(alpha_dir, "alpha_search_slope_center_heatmap.png"))

    # Top candidates.
    top_alpha = alpha_df.sort_values(["selection_score", "student_test_RMSE"], ascending=[False, True]).head(15).copy()
    top_alpha["Candidate"] = [f"A{i+1}" for i in range(len(top_alpha))]
    top_alpha.to_excel(os.path.join(alpha_dir, "top15_alpha_candidates.xlsx"), index=False)
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=top_alpha, x="selection_score", y="Candidate", edgecolor="black")
    ax.set_xlabel("Selection score", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title("Top Alpha Candidates", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="x")
    save_current_figure(os.path.join(alpha_dir, "top15_alpha_candidates.png"))

    return outputs



def gaussian_kernel_smooth_with_ci(x, y, grid=None, bandwidth=None, n_boot=300, ci=95, random_state=RANDOM_STATE):
    """
    Kernel-smoothed mean response with bootstrap confidence interval.

    This helper is used only for visualization. It does not change model
    training, model selection, alpha selection, or prediction logic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if grid is None:
        if len(np.unique(x)) >= 2:
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        else:
            grid = np.repeat(float(x[0]), 1)
    else:
        grid = np.asarray(grid, dtype=float)

    if bandwidth is None:
        x_range = float(np.max(x) - np.min(x))
        if x_range <= 1e-12:
            bandwidth = 1.0
        else:
            bandwidth = max(x_range * 0.12, np.std(x) * 0.25, 1e-6)

    def smooth_once(xx, yy):
        pred = np.zeros(len(grid), dtype=float)
        for i, gx in enumerate(grid):
            w = np.exp(-0.5 * ((xx - gx) / bandwidth) ** 2)
            w_sum = np.sum(w)
            if w_sum <= 1e-12:
                pred[i] = np.nan
            else:
                pred[i] = np.sum(w * yy) / w_sum
        return pred

    mean_line = smooth_once(x, y)

    if len(x) < 8 or n_boot <= 0:
        return grid, mean_line, np.full_like(mean_line, np.nan), np.full_like(mean_line, np.nan)

    rng = np.random.default_rng(random_state)
    boot_lines = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boot_lines.append(smooth_once(x[idx], y[idx]))
    boot_lines = np.asarray(boot_lines, dtype=float)

    alpha = (100 - ci) / 2
    lower = np.nanpercentile(boot_lines, alpha, axis=0)
    upper = np.nanpercentile(boot_lines, 100 - alpha, axis=0)
    return grid, mean_line, lower, upper


def plot_smooth_curve_with_ci(ax, x, y, label, color, linewidth=2.5, alpha_fill=0.18,
                              n_boot=300, bandwidth=None, scatter=False, scatter_kwargs=None):
    grid, mean_line, lower, upper = gaussian_kernel_smooth_with_ci(
        x=x,
        y=y,
        bandwidth=bandwidth,
        n_boot=n_boot,
        random_state=RANDOM_STATE,
    )

    if scatter:
        scatter_kwargs = scatter_kwargs or {}
        ax.scatter(x, y, **scatter_kwargs)

    if len(grid) > 0:
        ax.plot(grid, mean_line, color=color, linewidth=linewidth, label=label)
        if np.any(np.isfinite(lower)) and np.any(np.isfinite(upper)):
            ax.fill_between(grid, lower, upper, color=color, alpha=alpha_fill, linewidth=0)

    return ax


def plot_prediction_learning_effects(best_bundle, output_dir):
    """
    Plot sample-level learning-effect diagnostics for the selected model.
    These figures quantify how M3 changes M0 predictions and residuals.
    """
    outputs = {}
    if best_bundle is None:
        return outputs

    effect_dir = safe_mkdir(os.path.join(output_dir, "learning_effect_figures"))
    test_df = best_bundle["test_df"].copy()
    y_test = np.asarray(best_bundle["y_test"], dtype=float)
    m0_pred = np.asarray(best_bundle["m0_test_pred"], dtype=float)
    m3_pred = np.asarray(best_bundle["m2_test_pred"], dtype=float)
    raw_corr = np.asarray(best_bundle["raw_test_correction"], dtype=float)
    alpha = np.asarray(best_bundle["test_alpha"], dtype=float)
    applied_corr = alpha * raw_corr

    effect_df = test_df.copy()
    effect_df["True"] = y_test
    effect_df["M0 prediction"] = m0_pred
    effect_df[f"{MODEL_DISPLAY_NAME} prediction"] = m3_pred
    effect_df["Raw correction"] = raw_corr
    effect_df["Alpha"] = alpha
    effect_df["Applied correction"] = applied_corr
    effect_df["M0 residual"] = y_test - m0_pred
    effect_df[f"{MODEL_DISPLAY_NAME} residual"] = y_test - m3_pred
    effect_df["Absolute residual reduction"] = np.abs(effect_df["M0 residual"]) - np.abs(effect_df[f"{MODEL_DISPLAY_NAME} residual"])
    effect_df["Squared residual reduction"] = effect_df["M0 residual"] ** 2 - effect_df[f"{MODEL_DISPLAY_NAME} residual"] ** 2
    effect_df.to_excel(os.path.join(effect_dir, "selected_model_learning_effect_by_sample.xlsx"), index=False)
    outputs["sample_learning_effect"] = effect_df

    # Residual mapping from M0 to M3.
    plt.figure(figsize=FIGSIZE["correction_diagnostic"])
    ax = sns.scatterplot(data=effect_df, x="M0 residual", y=f"{MODEL_DISPLAY_NAME} residual", s=100, alpha=0.75, edgecolor="k", color="#b4d4e1")
    lim_min = float(min(effect_df["M0 residual"].min(), effect_df[f"{MODEL_DISPLAY_NAME} residual"].min()))
    lim_max = float(max(effect_df["M0 residual"].max(), effect_df[f"{MODEL_DISPLAY_NAME} residual"].max()))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1.5)
    ax.axhline(0, color="r", linestyle="--", linewidth=1.2)
    ax.axvline(0, color="r", linestyle="--", linewidth=1.2)
    ax.set_xlabel("M0 residual", fontsize=18, fontweight="bold")
    ax.set_ylabel(f"{MODEL_DISPLAY_NAME} residual", fontsize=18, fontweight="bold")
    ax.set_title("Residual Mapping", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(effect_dir, "residual_mapping_M0_to_M3.png"))

    # Residual reduction distribution.
    plt.figure(figsize=FIGSIZE["residual_distribution"])
    ax = sns.histplot(effect_df["Absolute residual reduction"], bins=20, kde=True, alpha=0.65, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Absolute residual reduction", fontsize=18, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=18, fontweight="bold")
    ax.set_title("Error Reduction Distribution", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(effect_dir, "absolute_residual_reduction_distribution.png"))

    # Prediction profile, sorted by true value.
    # True values are shown as observed points and a thin line, while M0 and M3
    # are shown as kernel-smoothed curves with bootstrap confidence intervals.
    profile_df = effect_df.sort_values("True").reset_index(drop=True)
    profile_df["Sample order"] = np.arange(1, len(profile_df) + 1)

    prediction_profile_data = profile_df[[
        "Sample order",
        "True",
        "M0 prediction",
        f"{MODEL_DISPLAY_NAME} prediction",
    ]].copy()
    prediction_profile_data.to_excel(os.path.join(effect_dir, "prediction_profile_smoothed_data.xlsx"), index=False)
    outputs["prediction_profile"] = prediction_profile_data

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(
        profile_df["Sample order"],
        profile_df["True"],
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.65,
        label="True",
    )
    ax.scatter(
        profile_df["Sample order"],
        profile_df["True"],
        color="#1f77b4",
        s=18,
        alpha=0.65,
        edgecolors="none",
    )

    plot_smooth_curve_with_ci(
        ax,
        profile_df["Sample order"].values,
        profile_df["M0 prediction"].values,
        label="M0",
        color="#ff7f0e",
        linewidth=2.8,
        alpha_fill=0.20,
        n_boot=300,
    )
    plot_smooth_curve_with_ci(
        ax,
        profile_df["Sample order"].values,
        profile_df[f"{MODEL_DISPLAY_NAME} prediction"].values,
        label=MODEL_DISPLAY_NAME,
        color="#2ca02c",
        linewidth=2.8,
        alpha_fill=0.20,
        n_boot=300,
    )

    ax.set_xlabel("Samples sorted by true value", fontsize=14, fontweight="bold")
    ax.set_ylabel("Chalkiness degree", fontsize=14, fontweight="bold")
    ax.set_title("Prediction Profile", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    ax.legend(fontsize=12)
    save_current_figure(os.path.join(effect_dir, "prediction_profile_sorted_by_true_value.png"))
    save_current_figure(os.path.join(effect_dir, "prediction_profile_smoothed_with_CI.png")) if False else None

    # Additional saved copy with an explicit filename for the smoothed version.
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(
        profile_df["Sample order"],
        profile_df["True"],
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.65,
        label="True",
    )
    ax.scatter(
        profile_df["Sample order"],
        profile_df["True"],
        color="#1f77b4",
        s=18,
        alpha=0.65,
        edgecolors="none",
    )
    plot_smooth_curve_with_ci(
        ax,
        profile_df["Sample order"].values,
        profile_df["M0 prediction"].values,
        label="M0",
        color="#ff7f0e",
        linewidth=2.8,
        alpha_fill=0.20,
        n_boot=300,
    )
    plot_smooth_curve_with_ci(
        ax,
        profile_df["Sample order"].values,
        profile_df[f"{MODEL_DISPLAY_NAME} prediction"].values,
        label=MODEL_DISPLAY_NAME,
        color="#2ca02c",
        linewidth=2.8,
        alpha_fill=0.20,
        n_boot=300,
    )
    ax.set_xlabel("Samples sorted by true value", fontsize=14, fontweight="bold")
    ax.set_ylabel("Chalkiness degree", fontsize=14, fontweight="bold")
    ax.set_title("Prediction Profile", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    ax.legend(fontsize=12)
    save_current_figure(os.path.join(effect_dir, "prediction_profile_smoothed_with_CI.png"))

    # Error reduction smooth distribution by observed chalkiness range.
    # This is not grouped by risk classes. It shows the smoothed conditional
    # mean of error reduction along the observed chalkiness gradient.
    x_chalk = effect_df["True"].values
    y_reduction = effect_df["Absolute residual reduction"].values
    grid, mean_line, lower, upper = gaussian_kernel_smooth_with_ci(
        x=x_chalk,
        y=y_reduction,
        n_boot=300,
        random_state=RANDOM_STATE,
    )
    error_reduction_chalkiness_df = pd.DataFrame({
        "Chalkiness degree": grid,
        "Smoothed absolute residual reduction": mean_line,
        "CI lower": lower,
        "CI upper": upper,
    })
    error_reduction_chalkiness_df.to_excel(
        os.path.join(effect_dir, "error_reduction_by_chalkiness_smooth_curve_data.xlsx"),
        index=False,
    )
    outputs["error_reduction_by_chalkiness"] = error_reduction_chalkiness_df

    plt.figure(figsize=FIGSIZE["correction_diagnostic"])
    ax = sns.scatterplot(
        data=effect_df,
        x="True",
        y="Absolute residual reduction",
        s=80,
        alpha=0.45,
        edgecolor="k",
        color="#b4d4e1",
    )
    if len(grid) > 0:
        ax.plot(grid, mean_line, color="blue", linewidth=3, label="Smoothed mean")
        if np.any(np.isfinite(lower)) and np.any(np.isfinite(upper)):
            ax.fill_between(grid, lower, upper, color="blue", alpha=0.18, linewidth=0, label="95% CI")
    ax.axhline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Chalkiness degree", fontsize=18, fontweight="bold")
    ax.set_ylabel("Absolute residual reduction", fontsize=18, fontweight="bold")
    ax.set_title("Error Reduction by Chalkiness", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    ax.legend(fontsize=12)
    save_current_figure(os.path.join(effect_dir, "error_reduction_by_chalkiness_smooth_curve.png"))

    # Relationship between mechanism variables and error reduction.
    for x_col in [TMIN_COL, TMIN_EXCESS_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL]:
        if x_col not in effect_df.columns:
            continue
        plt.figure(figsize=FIGSIZE["correction_diagnostic"])
        ax = sns.scatterplot(data=effect_df, x=x_col, y="Absolute residual reduction", s=100, alpha=0.75, edgecolor="k", color="#b4d4e1")
        sns.regplot(data=effect_df, x=x_col, y="Absolute residual reduction", scatter=False, ax=ax, color="blue")
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel(x_col, fontsize=18, fontweight="bold")
        ax.set_ylabel("Absolute residual reduction", fontsize=18, fontweight="bold")
        ax.set_title("Error Reduction", fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        safe_x = str(x_col).replace("/", "_").replace(" ", "_").replace("≥", "ge")
        save_current_figure(os.path.join(effect_dir, f"error_reduction_vs_{safe_x}.png"))

    # Correction decomposition.
    plt.figure(figsize=FIGSIZE["correction_diagnostic"])
    ax = sns.scatterplot(data=effect_df, x="Raw correction", y="Applied correction", hue="Alpha", s=100, alpha=0.75, edgecolor="k")
    ax.axhline(0, color="r", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="r", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Raw correction", fontsize=18, fontweight="bold")
    ax.set_ylabel("Applied correction", fontsize=18, fontweight="bold")
    ax.set_title("Correction Scaling", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(effect_dir, "raw_vs_applied_correction.png"))

    return outputs


def plot_teacher_residual_learning_diagnostics(best_bundle, output_dir):
    """
    Plot how the residual-correction target is distributed in the teacher data.
    This helps explain what the knowledge-constrained residual module learns.
    """
    outputs = {}
    if best_bundle is None or "teacher_residual_df" not in best_bundle:
        return outputs

    teacher_resid_df = best_bundle["teacher_residual_df"].copy()
    if teacher_resid_df.empty or "M0 residual on teacher" not in teacher_resid_df.columns:
        return outputs

    target_dir = safe_mkdir(os.path.join(output_dir, "learning_process_figures"))
    teacher_resid_df.to_excel(os.path.join(target_dir, "teacher_residual_learning_target.xlsx"), index=False)
    outputs["teacher_residual_target"] = teacher_resid_df

    plt.figure(figsize=FIGSIZE["residual_distribution"])
    ax = sns.histplot(teacher_resid_df["M0 residual on teacher"], bins=15, kde=True, alpha=0.65, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Teacher M0 residual", fontsize=18, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=18, fontweight="bold")
    ax.set_title("Residual Learning Target", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(target_dir, "teacher_residual_target_distribution.png"))

    for x_col in [TMIN_COL, TMIN_EXCESS_COL, PROTEIN_COL, PROTEIN_RISK_COL]:
        if x_col not in teacher_resid_df.columns:
            continue
        plt.figure(figsize=FIGSIZE["correction_diagnostic"])
        ax = sns.scatterplot(data=teacher_resid_df, x=x_col, y="M0 residual on teacher", s=100, alpha=0.75, edgecolor="k", color="#b4d4e1")
        sns.regplot(data=teacher_resid_df, x=x_col, y="M0 residual on teacher", scatter=False, ax=ax, color="blue")
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel(x_col, fontsize=18, fontweight="bold")
        ax.set_ylabel("Teacher M0 residual", fontsize=18, fontweight="bold")
        ax.set_title("Residual Target", fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        safe_x = str(x_col).replace("/", "_").replace(" ", "_").replace("≥", "ge")
        save_current_figure(os.path.join(target_dir, f"teacher_residual_target_vs_{safe_x}.png"))

    return outputs


def save_learning_figure_workbook(screen_outputs, alpha_outputs, effect_outputs, teacher_outputs, output_dir):
    workbook_path = os.path.join(output_dir, "M3_learning_process_and_effect_summary.xlsx")
    with pd.ExcelWriter(workbook_path) as writer:
        for source_dict in [screen_outputs, alpha_outputs, effect_outputs, teacher_outputs]:
            for key, value in source_dict.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=str(key)[:31], index=False)
    return workbook_path



# =========================================================
# 5. Repeated independent training workflow
# =========================================================
# User-requested screening settings.
N_INDEPENDENT_TRAININGS = 50
TRAIN_FRACTIONS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
FIXED_TEST_FRACTION = 0.30

# Folder names for structured outputs.
SCREENING_ROOT_NAME = "M3_repeated_independent_training_screening"
FOLDER_METADATA = "00_metadata"
FOLDER_ALL_RESULTS = "01_all_results"
FOLDER_PER_RATIO = "02_by_training_fraction"
FOLDER_BOXPLOTS = "03_repeated_training_boxplots"
FOLDER_FIXED_MODEL = "04_fixed_model"
FOLDER_MECHANISM = "05_mechanism_validation"
FOLDER_SCATTER = "06_fixed_model_prediction_scatter"


def safe_name(x):
    s = str(x)
    for old, new in [(" ", "_"), ("/", "_"), ("\\\\", "_"), (":", "_"), (";", "_"), (".", "p")]:
        s = s.replace(old, new)
    return s


def make_fixed_spec_key(rec):
    return (
        float(rec.get("training_fraction", np.nan)),
        str(rec.get("Model family", "")),
        str(rec.get("Version", "")),
        str(rec.get("Weight scheme", "none")),
        float(rec.get("Weight beta", 0.0)) if pd.notna(rec.get("Weight beta", np.nan)) else 0.0,
        str(rec.get("Weight location", "none")),
        int(rec.get("z_score_threshold", -1)),
        int(rec.get("cv", -1)),
    )


def enrich_record_for_repeated_screening(rec, training_fraction, test_fraction, unused_fraction,
                                         iteration, split_random_state, train_idx, test_idx, unused_idx):
    rec.update({
        "training_iteration": int(iteration),
        "training_fraction": float(training_fraction),
        "test_fraction": float(test_fraction),
        "unused_fraction": float(unused_fraction),
        "split_random_state": int(split_random_state),
        "n_unused": int(len(unused_idx)),
        "train_indices": json.dumps([int(i) for i in train_idx]),
        "test_indices": json.dumps([int(i) for i in test_idx]),
        "unused_indices": json.dumps([int(i) for i in unused_idx]),
    })
    return rec


def make_train_test_unused_split(n_samples, training_fraction, test_fraction, random_state):
    if training_fraction + test_fraction > 1.0 + 1e-12:
        raise ValueError("training_fraction + test_fraction must be <= 1.")

    all_idx = np.arange(n_samples)
    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_fraction,
        random_state=random_state,
        shuffle=True,
    )

    train_n = int(round(training_fraction * n_samples))
    train_n = max(1, min(train_n, len(dev_idx)))

    rng = np.random.default_rng(random_state)
    train_idx = np.sort(rng.choice(dev_idx, size=train_n, replace=False))
    unused_idx = np.sort(np.setdiff1d(dev_idx, train_idx))
    test_idx = np.sort(test_idx)
    return train_idx, test_idx, unused_idx


def write_iteration_outputs(iteration_records, train_fraction_dir, iteration):
    iter_dir = safe_mkdir(os.path.join(train_fraction_dir, f"iteration_{iteration:03d}"))
    iter_df = pd.DataFrame(iteration_records)
    iter_df.to_excel(os.path.join(iter_dir, f"iteration_{iteration:03d}_all_model_results.xlsx"), index=False)

    core_cols = [
        "training_iteration", "training_fraction", "test_fraction", "unused_fraction",
        "Model family", "Version", "Weight scheme", "Weight beta", "Weight location",
        "z_score_threshold", "cv", "n_train", "n_test", "n_unused",
        "M0 test R2", "M0 test RMSE", "M0 test MAE",
        "WR0 M2 test R2", "R5C M2 test R2",
        "M2 test R2", "M2 test RMSE", "M2 test MAE",
        "Delta test R2 M2_minus_M0", "Delta test R2 M2_minus_WR0", "Delta test R2 M2_minus_R5C",
        "Delta test RMSE M2_minus_M0", "Delta test RMSE M2_minus_WR0", "Delta test RMSE M2_minus_R5C",
        "M0 overfit gap", "M2 overfit gap",
        "M0_best_params", "Protein_proxy_best_params",
        "Smooth_alpha_base", "Smooth_alpha_amp", "Smooth_alpha_slope", "Smooth_alpha_center_quantile",
        "Smooth_alpha_center", "Smooth_alpha_scale", "Smooth_alpha_direction",
    ]
    core_cols = [c for c in core_cols if c in iter_df.columns]
    iter_df[core_cols].to_excel(os.path.join(iter_dir, f"iteration_{iteration:03d}_core_metrics.xlsx"), index=False)
    return iter_dir


def update_best_bundle_by_spec(best_bundle_by_spec, rec, bundle):
    key = make_fixed_spec_key(rec)
    old = best_bundle_by_spec.get(key)
    if old is None:
        best_bundle_by_spec[key] = {"record": rec.copy(), "bundle": bundle.copy()}
        return
    old_rec = old["record"]
    if (rec["M2 test R2"] > old_rec["M2 test R2"]) or (
        np.isclose(rec["M2 test R2"], old_rec["M2 test R2"]) and rec["M2 test RMSE"] < old_rec["M2 test RMSE"]
    ):
        best_bundle_by_spec[key] = {"record": rec.copy(), "bundle": bundle.copy()}


def run_one_independent_training(student_clean, teacher_clean, base_weather_features, original_target_col,
                                 protein_info, protein_oof_metrics, z_thr, cv, training_fraction,
                                 iteration, split_random_state, train_fraction_dir, best_bundle_by_spec):
    n_samples = len(student_clean)
    train_idx, test_idx, unused_idx = make_train_test_unused_split(
        n_samples=n_samples,
        training_fraction=training_fraction,
        test_fraction=FIXED_TEST_FRACTION,
        random_state=split_random_state,
    )
    unused_fraction = len(unused_idx) / n_samples

    train_df = student_clean.iloc[train_idx].copy()
    test_df = student_clean.iloc[test_idx].copy()
    X_train = train_df[base_weather_features].values
    X_test = test_df[base_weather_features].values
    y_all = student_clean[original_target_col].values
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    if cv >= len(X_train):
        print(f"Iteration {iteration:03d}, train fraction {training_fraction:.2f}: CV={cv} >= n_train={len(X_train)}; skipped.")
        return []

    iteration_records = []

    print(
        f"\nTraining iteration {iteration:03d} | train={training_fraction:.0%} | "
        f"test={FIXED_TEST_FRACTION:.0%} | unused={unused_fraction:.0%} | n_train={len(train_idx)} | n_test={len(test_idx)}"
    )

    m0_model, m0_scaler, m0_search, m0_train_pred, m0_test_pred = train_m0_rf_on_fixed_split(
        X_train,
        y_train,
        X_test,
        cv,
    )
    m0_train_metrics = calculate_metrics(y_train, m0_train_pred, "M0 train")
    m0_test_metrics = calculate_metrics(y_test, m0_test_pred, "M0 test")

    # ---------------------------------------------------------
    # WR0 baseline.
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
        student_outliers=[],
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
    wr0_rec["Delta test RMSE M2_minus_R5C"] = np.nan
    wr0_rec["Delta test MAE M2_minus_R5C"] = np.nan
    wr0_rec = enrich_record_for_repeated_screening(
        wr0_rec, training_fraction, FIXED_TEST_FRACTION, unused_fraction,
        iteration, split_random_state, train_idx, test_idx, unused_idx,
    )
    iteration_records.append(wr0_rec)
    wr0_full_bundle = {
        "model_kind": "WR0",
        "student_clean": student_clean.copy(),
        "train_df": train_df.copy(),
        "test_df": test_df.copy(),
        "unused_df": student_clean.iloc[unused_idx].copy(),
        "m0_model": m0_model,
        "m0_scaler": m0_scaler,
        "m0_train_pred": m0_train_pred,
        "m0_test_pred": m0_test_pred,
        "residual_model": wr0_model,
        "residual_scaler": wr0_scaler,
        "residual_feature_cols": wr0_feature_cols,
        "raw_train_correction": wr0_raw_train,
        "raw_test_correction": wr0_raw_test,
        "alpha_info": wr0_bundle["alpha_info"],
        "train_alpha": wr0_bundle["train_alpha"],
        "test_alpha": wr0_bundle["test_alpha"],
        "alpha_df": wr0_bundle["alpha_df"],
        "m2_train_pred": wr0_bundle["m2_train_pred"],
        "m2_test_pred": wr0_bundle["m2_test_pred"],
        "teacher_residual_df": wr0_teacher_df.copy(),
        "y_train": y_train,
        "y_test": y_test,
    }
    update_best_bundle_by_spec(best_bundle_by_spec, wr0_rec, wr0_full_bundle)

    # ---------------------------------------------------------
    # R5C baseline.
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
        student_outliers=[],
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
    r5_rec["Delta test RMSE M2_minus_R5C"] = 0.0
    r5_rec["Delta test MAE M2_minus_R5C"] = 0.0
    r5_rec = enrich_record_for_repeated_screening(
        r5_rec, training_fraction, FIXED_TEST_FRACTION, unused_fraction,
        iteration, split_random_state, train_idx, test_idx, unused_idx,
    )
    iteration_records.append(r5_rec)
    r5_full_bundle = {
        "model_kind": "R5C",
        "student_clean": student_clean.copy(),
        "train_df": train_df.copy(),
        "test_df": test_df.copy(),
        "unused_df": student_clean.iloc[unused_idx].copy(),
        "m0_model": m0_model,
        "m0_scaler": m0_scaler,
        "m0_train_pred": m0_train_pred,
        "m0_test_pred": m0_test_pred,
        "residual_model": r5_model,
        "residual_scaler": None,
        "residual_feature_cols": r5_feature_cols,
        "raw_train_correction": r5_raw_train,
        "raw_test_correction": r5_raw_test,
        "alpha_info": r5_bundle["alpha_info"],
        "train_alpha": r5_bundle["train_alpha"],
        "test_alpha": r5_bundle["test_alpha"],
        "alpha_df": r5_bundle["alpha_df"],
        "m2_train_pred": r5_bundle["m2_train_pred"],
        "m2_test_pred": r5_bundle["m2_test_pred"],
        "teacher_residual_df": r5_teacher_df.copy(),
        "y_train": y_train,
        "y_test": y_test,
    }
    update_best_bundle_by_spec(best_bundle_by_spec, r5_rec, r5_full_bundle)

    # ---------------------------------------------------------
    # R5CW: conditional constraint + sample weighting.
    # ---------------------------------------------------------
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

    for spec in weighted_specs:
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
            student_outliers=[],
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
        cw_rec = enrich_record_for_repeated_screening(
            cw_rec, training_fraction, FIXED_TEST_FRACTION, unused_fraction,
            iteration, split_random_state, train_idx, test_idx, unused_idx,
        )
        iteration_records.append(cw_rec)

        cw_full_bundle = {
            "model_kind": "R5CW",
            "student_clean": student_clean.copy(),
            "train_df": train_df.copy(),
            "test_df": test_df.copy(),
            "unused_df": student_clean.iloc[unused_idx].copy(),
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
            "alpha_df": cw_bundle["alpha_df"],
            "m2_train_pred": cw_bundle["m2_train_pred"],
            "m2_test_pred": cw_bundle["m2_test_pred"],
            "teacher_residual_df": cw_teacher_df.copy(),
            "y_train": y_train,
            "y_test": y_test,
        }
        update_best_bundle_by_spec(best_bundle_by_spec, cw_rec, cw_full_bundle)

    write_iteration_outputs(iteration_records, train_fraction_dir, iteration)
    return iteration_records


def summarize_repeated_results(result_df, output_dir):
    summary_cols = [
        "training_fraction", "Model family", "Version", "Weight scheme", "Weight beta", "Weight location",
        "z_score_threshold", "cv",
    ]
    metric_cols = [
        "M2 test R2", "M2 test RMSE", "M2 test MAE",
        "M0 test R2", "M0 test RMSE", "M0 test MAE",
        "Delta test R2 M2_minus_M0", "Delta test R2 M2_minus_WR0", "Delta test R2 M2_minus_R5C",
        "Delta test RMSE M2_minus_M0", "Delta test RMSE M2_minus_WR0", "Delta test RMSE M2_minus_R5C",
        "M0 overfit gap", "M2 overfit gap",
    ]
    metric_cols = [c for c in metric_cols if c in result_df.columns]
    agg_dict = {}
    for c in metric_cols:
        agg_dict[f"{c} mean"] = (c, "mean")
        agg_dict[f"{c} std"] = (c, "std")
        agg_dict[f"{c} median"] = (c, "median")
        agg_dict[f"{c} min"] = (c, "min")
        agg_dict[f"{c} max"] = (c, "max")
    summary = result_df.groupby(summary_cols, dropna=False).agg(
        n_runs=("training_iteration", "count"),
        **agg_dict,
    ).reset_index()
    summary = summary.sort_values(["M2 test R2 mean", "M2 test RMSE mean"], ascending=[False, True])
    summary.to_excel(os.path.join(output_dir, "repeated_training_grouped_summary.xlsx"), index=False)

    family_summary = result_df.groupby(["training_fraction", "Model family"], dropna=False).agg(
        n_runs=("training_iteration", "count"),
        R2_mean=("M2 test R2", "mean"),
        R2_std=("M2 test R2", "std"),
        R2_median=("M2 test R2", "median"),
        RMSE_mean=("M2 test RMSE", "mean"),
        RMSE_std=("M2 test RMSE", "std"),
        MAE_mean=("M2 test MAE", "mean"),
        MAE_std=("M2 test MAE", "std"),
    ).reset_index()
    family_summary.to_excel(os.path.join(output_dir, "repeated_training_family_summary.xlsx"), index=False)
    return summary, family_summary


def plot_repeated_training_boxplots(result_df, output_dir):
    box_dir = safe_mkdir(os.path.join(output_dir, FOLDER_BOXPLOTS))
    df = result_df.copy()
    df["training_fraction_label"] = (df["training_fraction"] * 100).round().astype(int).astype(str) + "%"

    plot_specs = [
        ("M2 test R2", "Validation R²", "boxplot_validation_R2_by_train_fraction_and_model.png"),
        ("M2 test RMSE", "Validation RMSE", "boxplot_validation_RMSE_by_train_fraction_and_model.png"),
        ("M2 test MAE", "Validation MAE", "boxplot_validation_MAE_by_train_fraction_and_model.png"),
        ("Delta test R2 M2_minus_M0", "Delta R² vs M0", "boxplot_delta_R2_vs_M0_by_train_fraction_and_model.png"),
        ("Delta test R2 M2_minus_WR0", "Delta R² vs WR0", "boxplot_delta_R2_vs_WR0_by_train_fraction_and_model.png"),
        ("Delta test R2 M2_minus_R5C", "Delta R² vs R5C", "boxplot_delta_R2_vs_R5C_by_train_fraction_and_model.png"),
    ]

    for y_col, ylabel, fname in plot_specs:
        if y_col not in df.columns:
            continue
        plot_df = df.dropna(subset=[y_col]).copy()
        if plot_df.empty:
            continue
        plt.figure(figsize=(10, 5.8))
        ax = sns.boxplot(data=plot_df, x="training_fraction_label", y=y_col, hue="Model family")
        ax.set_xlabel("Training fraction", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(ylabel, fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        ax.legend(title="Model family", fontsize=10, title_fontsize=11)
        save_current_figure(os.path.join(box_dir, fname))

    # Boxplots only for final R5CW candidates.
    r5cw = df[df["Model family"] == "R5CW"].copy()
    if not r5cw.empty:
        plt.figure(figsize=(10, 5.8))
        ax = sns.boxplot(data=r5cw, x="training_fraction_label", y="M2 test R2")
        ax.set_xlabel("Training fraction", fontsize=14, fontweight="bold")
        ax.set_ylabel("R5CW validation R²", fontsize=14, fontweight="bold")
        ax.set_title("R5CW Stability across Training Fractions", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        save_current_figure(os.path.join(box_dir, "boxplot_R5CW_validation_R2_by_train_fraction.png"))

    return box_dir


def select_fixed_model_from_repeated_results(summary_df, best_bundle_by_spec, output_dir):
    fixed_dir = safe_mkdir(os.path.join(output_dir, FOLDER_FIXED_MODEL))
    r5cw_summary = summary_df[summary_df["Model family"] == "R5CW"].copy()
    if r5cw_summary.empty:
        raise RuntimeError("No R5CW model is available for fixed-model selection.")

    r5cw_summary = r5cw_summary.sort_values(
        ["M2 test R2 mean", "M2 test RMSE mean", "M2 test R2 std"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    r5cw_summary.to_excel(os.path.join(fixed_dir, "fixed_model_selection_candidates_R5CW.xlsx"), index=False)
    selected_summary = r5cw_summary.iloc[0].copy()

    selected_key = (
        float(selected_summary["training_fraction"]),
        str(selected_summary["Model family"]),
        str(selected_summary["Version"]),
        str(selected_summary["Weight scheme"]),
        float(selected_summary["Weight beta"]) if pd.notna(selected_summary["Weight beta"]) else 0.0,
        str(selected_summary["Weight location"]),
        int(selected_summary["z_score_threshold"]),
        int(selected_summary["cv"]),
    )

    selected_bundle_obj = best_bundle_by_spec.get(selected_key)
    if selected_bundle_obj is None:
        raise RuntimeError("The selected fixed-model specification has no saved fitted bundle.")

    selected_record = selected_bundle_obj["record"].copy()
    selected_bundle = selected_bundle_obj["bundle"].copy()

    pd.DataFrame([selected_summary]).to_excel(os.path.join(fixed_dir, "fixed_model_selected_group_mean_summary.xlsx"), index=False)
    pd.DataFrame([selected_record]).to_excel(os.path.join(fixed_dir, "fixed_model_selected_fitted_run_metrics.xlsx"), index=False)

    fixed_params = {
        "selection_rule": "Highest mean validation R2 across 50 independent trainings, with lower mean RMSE and lower R2 SD as tie-breakers.",
        "selected_training_fraction": float(selected_summary["training_fraction"]),
        "selected_model_family": str(selected_summary["Model family"]),
        "selected_version": str(selected_summary["Version"]),
        "selected_weight_scheme": str(selected_summary["Weight scheme"]),
        "selected_weight_beta": float(selected_summary["Weight beta"]) if pd.notna(selected_summary["Weight beta"]) else None,
        "selected_weight_location": str(selected_summary["Weight location"]),
        "selected_z_score_threshold": int(selected_summary["z_score_threshold"]),
        "selected_cv": int(selected_summary["cv"]),
        "selected_mean_validation_R2": float(selected_summary["M2 test R2 mean"]),
        "selected_mean_validation_RMSE": float(selected_summary["M2 test RMSE mean"]),
        "selected_fitted_run_iteration": int(selected_record["training_iteration"]),
        "selected_fitted_run_validation_R2": float(selected_record["M2 test R2"]),
        "M0_best_params": selected_record.get("M0_best_params"),
        "Protein_proxy_best_params": selected_record.get("Protein_proxy_best_params"),
        "alpha_info": selected_bundle["alpha_info"],
        "residual_feature_cols": selected_bundle.get("residual_feature_cols"),
    }
    with open(os.path.join(fixed_dir, "fixed_model_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(fixed_params, f, ensure_ascii=False, indent=2)

    joblib.dump(selected_bundle["m0_model"], os.path.join(fixed_dir, "fixed_M3_M0_model.pkl"))
    joblib.dump(selected_bundle["m0_scaler"], os.path.join(fixed_dir, "fixed_M3_M0_scaler.pkl"))
    joblib.dump(selected_bundle["residual_model"], os.path.join(fixed_dir, "fixed_M3_conditional_residual_model.pkl"))
    joblib.dump(selected_bundle["alpha_info"], os.path.join(fixed_dir, "fixed_M3_alpha_info.pkl"))

    return selected_summary, selected_record, selected_bundle, fixed_dir


def save_fixed_model_predictions(selected_bundle, selected_record, output_dir, original_target_col):
    fixed_dir = safe_mkdir(os.path.join(output_dir, FOLDER_FIXED_MODEL))
    train_pred_df = selected_bundle["train_df"].copy()
    test_pred_df = selected_bundle["test_df"].copy()

    train_pred_df["M0 prediction"] = selected_bundle["m0_train_pred"]
    train_pred_df["Raw residual correction"] = selected_bundle["raw_train_correction"]
    train_pred_df["Adaptive alpha"] = selected_bundle["train_alpha"]
    train_pred_df["M3 prediction"] = selected_bundle["m2_train_pred"]
    train_pred_df["M0 residual"] = train_pred_df[original_target_col] - train_pred_df["M0 prediction"]
    train_pred_df["M3 residual"] = train_pred_df[original_target_col] - train_pred_df["M3 prediction"]

    test_pred_df["M0 prediction"] = selected_bundle["m0_test_pred"]
    test_pred_df["Raw residual correction"] = selected_bundle["raw_test_correction"]
    test_pred_df["Adaptive alpha"] = selected_bundle["test_alpha"]
    test_pred_df["M3 prediction"] = selected_bundle["m2_test_pred"]
    test_pred_df["M0 residual"] = test_pred_df[original_target_col] - test_pred_df["M0 prediction"]
    test_pred_df["M3 residual"] = test_pred_df[original_target_col] - test_pred_df["M3 prediction"]

    train_pred_df.to_excel(os.path.join(fixed_dir, "fixed_M3_train_predictions.xlsx"), index=False)
    test_pred_df.to_excel(os.path.join(fixed_dir, "fixed_M3_validation_predictions.xlsx"), index=False)
    return train_pred_df, test_pred_df


def plot_fixed_model_scatter(selected_bundle, output_dir):
    scatter_dir = safe_mkdir(os.path.join(output_dir, FOLDER_SCATTER))
    plot_true_vs_pred_extended(
        selected_bundle["y_train"],
        selected_bundle["m0_train_pred"],
        selected_bundle["y_test"],
        selected_bundle["m0_test_pred"],
        "M0 baseline for fixed M3 specification",
        scatter_dir,
        "fixed_M0",
    )
    plot_true_vs_pred_extended(
        selected_bundle["y_train"],
        selected_bundle["m2_train_pred"],
        selected_bundle["y_test"],
        selected_bundle["m2_test_pred"],
        "Fixed M3 model",
        scatter_dir,
        "fixed_M3",
    )
    return scatter_dir


def plot_pdp_monotonic_constraint_validation(selected_bundle, output_dir):
    mechanism_dir = safe_mkdir(os.path.join(output_dir, FOLDER_MECHANISM))
    residual_model = selected_bundle["residual_model"]
    feature_cols = selected_bundle["residual_feature_cols"]
    test_df = selected_bundle["test_df"].copy()
    feature_df = build_r5_residual_feature_df(test_df, [c for c in selected_bundle["student_clean"].columns if c in test_df.columns and c not in [TARGET_COL, ORIGINAL_TARGET_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL, TMIN_EXCESS_COL, TMIN_FLAG_COL]], PREDICTED_PROTEIN_COL) if False else None

    # Build the exact residual-model input from the selected fixed model.
    # The helper requires the same base weather columns that were used during training.
    base_weather_cols = [c for c in selected_bundle["student_clean"].columns if c in test_df.columns]
    excluded = set([ORIGINAL_TARGET_COL, TARGET_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL, TMIN_EXCESS_COL, TMIN_FLAG_COL])
    base_weather_cols = [c for c in base_weather_cols if c not in excluded]
    base_weather_cols = [c for c in base_weather_cols if np.issubdtype(test_df[c].dtype, np.number)]

    residual_feature_df = build_r5_residual_feature_df(test_df, base_weather_cols, PREDICTED_PROTEIN_COL)[feature_cols]
    high_df = residual_feature_df[residual_feature_df[TMIN_EXCESS_COL] > 0].copy()
    if high_df.empty or PROTEIN_RISK_COL not in high_df.columns:
        pd.DataFrame([{"message": "No high-TMIN samples or Protein_Heat_Risk missing; PDP monotonic validation was skipped."}]).to_excel(
            os.path.join(mechanism_dir, "PDP_monotonic_constraint_skipped.xlsx"), index=False
        )
        return None

    risk_values = high_df[PROTEIN_RISK_COL].values
    grid = np.linspace(float(np.nanpercentile(risk_values, 1)), float(np.nanpercentile(risk_values, 99)), 60)
    pdp_mean = []
    pdp_sd = []
    for value in grid:
        tmp = high_df.copy()
        tmp[PROTEIN_RISK_COL] = value
        pred = residual_model.predict(tmp[feature_cols].values)
        pdp_mean.append(float(np.mean(pred)))
        pdp_sd.append(float(np.std(pred)))

    pdp_mean = np.asarray(pdp_mean, dtype=float)
    pdp_sd = np.asarray(pdp_sd, dtype=float)
    diffs = np.diff(pdp_mean)
    violation_rate = float(np.mean(diffs < -1e-9)) if len(diffs) else np.nan

    pdp_df = pd.DataFrame({
        PROTEIN_RISK_COL: grid,
        "PDP residual correction mean": pdp_mean,
        "PDP residual correction SD": pdp_sd,
        "monotonic_violation_rate": violation_rate,
    })
    pdp_df.to_excel(os.path.join(mechanism_dir, "PDP_monotonic_constraint_Protein_Heat_Risk.xlsx"), index=False)

    plt.figure(figsize=FIGSIZE["correction_diagnostic"])
    ax = plt.gca()
    ax.plot(grid, pdp_mean, linewidth=3, label="PDP mean")
    ax.fill_between(grid, pdp_mean - pdp_sd, pdp_mean + pdp_sd, alpha=0.18, linewidth=0, label="±1 SD")
    ax.set_xlabel(PROTEIN_RISK_COL, fontsize=18, fontweight="bold")
    ax.set_ylabel("Residual correction", fontsize=18, fontweight="bold")
    ax.set_title("PDP Monotonic Constraint Validation", fontsize=20, fontweight="bold")
    ax.text(0.05, 0.95, f"Violation rate = {violation_rate:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    format_axes_code2(ax, grid_axis="both")
    ax.legend(fontsize=12)
    save_current_figure(os.path.join(mechanism_dir, "PDP_monotonic_constraint_Protein_Heat_Risk.png"))
    return pdp_df


def run_mechanism_validation_outputs(selected_bundle, selected_record, output_dir, base_weather_features):
    mechanism_dir = safe_mkdir(os.path.join(output_dir, FOLDER_MECHANISM))

    # 1. M0 and M3 residual distribution comparison.
    residual_diag_df = plot_residual_diagnostics(
        selected_bundle["y_test"],
        selected_bundle["m0_test_pred"],
        selected_bundle["m2_test_pred"],
        mechanism_dir,
    )

    # 2. Alpha-function rationality diagnostics.
    correction_bundle = selected_bundle.copy()
    correction_bundle["test_df"] = selected_bundle["test_df"].copy()
    correction_diag_df = plot_alpha_and_correction_diagnostics(correction_bundle, mechanism_dir)

    # 3. PDP monotonic-constraint validation.
    # Use explicit base weather columns for a robust residual input.
    residual_feature_df = build_r5_residual_feature_df(
        selected_bundle["test_df"], base_weather_features, PREDICTED_PROTEIN_COL
    )[selected_bundle["residual_feature_cols"]]
    high_df = residual_feature_df[residual_feature_df[TMIN_EXCESS_COL] > 0].copy()
    pdp_df = None
    if not high_df.empty and PROTEIN_RISK_COL in high_df.columns:
        risk_values = high_df[PROTEIN_RISK_COL].values
        grid = np.linspace(float(np.nanpercentile(risk_values, 1)), float(np.nanpercentile(risk_values, 99)), 60)
        pdp_mean = []
        pdp_sd = []
        for value in grid:
            tmp = high_df.copy()
            tmp[PROTEIN_RISK_COL] = value
            pred = selected_bundle["residual_model"].predict(tmp[selected_bundle["residual_feature_cols"]].values)
            pdp_mean.append(float(np.mean(pred)))
            pdp_sd.append(float(np.std(pred)))
        pdp_mean = np.asarray(pdp_mean, dtype=float)
        pdp_sd = np.asarray(pdp_sd, dtype=float)
        diffs = np.diff(pdp_mean)
        violation_rate = float(np.mean(diffs < -1e-9)) if len(diffs) else np.nan
        pdp_df = pd.DataFrame({
            PROTEIN_RISK_COL: grid,
            "PDP residual correction mean": pdp_mean,
            "PDP residual correction SD": pdp_sd,
            "monotonic_violation_rate": violation_rate,
        })
        pdp_df.to_excel(os.path.join(mechanism_dir, "PDP_monotonic_constraint_Protein_Heat_Risk.xlsx"), index=False)

        plt.figure(figsize=FIGSIZE["correction_diagnostic"])
        ax = plt.gca()
        ax.plot(grid, pdp_mean, linewidth=3, label="PDP mean")
        ax.fill_between(grid, pdp_mean - pdp_sd, pdp_mean + pdp_sd, alpha=0.18, linewidth=0, label="±1 SD")
        ax.set_xlabel(PROTEIN_RISK_COL, fontsize=18, fontweight="bold")
        ax.set_ylabel("Residual correction", fontsize=18, fontweight="bold")
        ax.set_title("PDP Monotonic Constraint Validation", fontsize=20, fontweight="bold")
        ax.text(0.05, 0.95, f"Violation rate = {violation_rate:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        format_axes_code2(ax, grid_axis="both")
        ax.legend(fontsize=12)
        save_current_figure(os.path.join(mechanism_dir, "PDP_monotonic_constraint_Protein_Heat_Risk.png"))
    else:
        pd.DataFrame([{"message": "No high-TMIN samples or Protein_Heat_Risk missing; PDP monotonic validation was skipped."}]).to_excel(
            os.path.join(mechanism_dir, "PDP_monotonic_constraint_skipped.xlsx"), index=False
        )

    # Workbook containing all mechanism-validation tables.
    workbook_path = os.path.join(mechanism_dir, "mechanism_validation_summary.xlsx")
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame([selected_record]).to_excel(writer, sheet_name="fixed_model_record", index=False)
        residual_diag_df.to_excel(writer, sheet_name="residual_diagnostics", index=False)
        correction_diag_df.to_excel(writer, sheet_name="alpha_correction", index=False)
        if pdp_df is not None:
            pdp_df.to_excel(writer, sheet_name="PDP_monotonic", index=False)
    return mechanism_dir


def main():
    screening_root = safe_mkdir(os.path.join(OUTPUT_DIR, SCREENING_ROOT_NAME))
    metadata_dir = safe_mkdir(os.path.join(screening_root, FOLDER_METADATA))
    all_results_dir = safe_mkdir(os.path.join(screening_root, FOLDER_ALL_RESULTS))
    per_ratio_root = safe_mkdir(os.path.join(screening_root, FOLDER_PER_RATIO))

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

    print("\nBase weather features for M0, protein proxy, and residual models:")
    print(base_weather_features)

    pd.DataFrame({"base_weather_features": base_weather_features}).to_excel(
        os.path.join(metadata_dir, "base_weather_features.xlsx"), index=False
    )

    config_df = pd.DataFrame([{
        "N_INDEPENDENT_TRAININGS": N_INDEPENDENT_TRAININGS,
        "TRAIN_FRACTIONS": json.dumps(TRAIN_FRACTIONS),
        "FIXED_TEST_FRACTION": FIXED_TEST_FRACTION,
        "STUDENT_Z_THRESHOLDS": json.dumps(STUDENT_Z_THRESHOLDS),
        "STUDENT_CV_VALUES": json.dumps(STUDENT_CV_VALUES),
        "WEIGHT_SCHEMES": json.dumps(WEIGHT_SCHEMES),
        "WEIGHT_BETAS": json.dumps(WEIGHT_BETAS),
        "WEIGHT_LOCATIONS": json.dumps(WEIGHT_LOCATIONS),
        "TMIN_THRESHOLD": TMIN_THRESHOLD,
        "note": "The 30% test split is used as the internal validation set for model, alpha, and weight selection. A separately held-out external dataset should be used for final independent validation.",
    }])
    config_df.to_excel(os.path.join(metadata_dir, "screening_configuration.xlsx"), index=False)

    teacher_df = teacher_numeric[[TARGET_COL, PROTEIN_COL] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    teacher_clean, teacher_outliers = remove_outliers_by_zscore(
        teacher_df,
        base_weather_features + [PROTEIN_COL],
        TEACHER_Z_THRESHOLD,
    )
    teacher_clean = add_protein_heat_risk(teacher_clean, PROTEIN_COL)

    pd.DataFrame({"teacher_outlier_index": teacher_outliers}).to_excel(
        os.path.join(metadata_dir, "teacher_removed_outliers.xlsx"), index=False
    )

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

    pd.DataFrame([{
        **protein_info,
        **protein_oof_metrics,
        "best_params_json": json.dumps(protein_info["best_params"], ensure_ascii=False),
    }]).to_excel(os.path.join(metadata_dir, "protein_proxy_training_summary.xlsx"), index=False)
    joblib.dump(protein_model, os.path.join(metadata_dir, "protein_proxy_model.pkl"))
    joblib.dump(protein_scaler, os.path.join(metadata_dir, "protein_proxy_scaler.pkl"))

    print("\nProtein proxy best params:", protein_info["best_params"])
    print("Protein proxy CV R2:", protein_info["best_repeated_cv_score"])
    print("Protein proxy OOF metrics:", protein_oof_metrics)

    student_df = original_numeric[[original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    student_df[PREDICTED_PROTEIN_COL] = protein_model.predict(protein_scaler.transform(student_df[base_weather_features].values))
    student_df = add_protein_heat_risk(student_df, PREDICTED_PROTEIN_COL)
    print("\nStudent valid n:", len(student_df))

    all_records = []
    best_bundle_by_spec = {}

    for z_thr in STUDENT_Z_THRESHOLDS:
        student_clean, student_outliers = remove_outliers_by_zscore(student_df, base_weather_features, z_thr)
        if len(student_clean) < 30:
            print(f"Z-score={z_thr}: too few samples after cleaning; skipped.")
            continue

        z_dir = safe_mkdir(os.path.join(per_ratio_root, f"zscore_{z_thr}"))
        pd.DataFrame({"student_outlier_index": student_outliers}).to_excel(
            os.path.join(z_dir, f"student_removed_outliers_zscore_{z_thr}.xlsx"), index=False
        )

        for cv in STUDENT_CV_VALUES:
            for training_fraction in TRAIN_FRACTIONS:
                train_fraction_label = f"train_{int(round(training_fraction * 100)):02d}_test_{int(round(FIXED_TEST_FRACTION * 100)):02d}_unused_{int(round((1 - training_fraction - FIXED_TEST_FRACTION) * 100)):02d}"
                train_fraction_dir = safe_mkdir(os.path.join(z_dir, f"cv_{cv}", train_fraction_label))
                ratio_records = []

                for iteration in range(1, N_INDEPENDENT_TRAININGS + 1):
                    split_random_state = RANDOM_STATE + int(training_fraction * 1000) + z_thr * 10000 + cv * 1000 + iteration
                    iteration_records = run_one_independent_training(
                        student_clean=student_clean,
                        teacher_clean=teacher_clean,
                        base_weather_features=base_weather_features,
                        original_target_col=original_target_col,
                        protein_info=protein_info,
                        protein_oof_metrics=protein_oof_metrics,
                        z_thr=z_thr,
                        cv=cv,
                        training_fraction=training_fraction,
                        iteration=iteration,
                        split_random_state=split_random_state,
                        train_fraction_dir=train_fraction_dir,
                        best_bundle_by_spec=best_bundle_by_spec,
                    )
                    ratio_records.extend(iteration_records)
                    all_records.extend(iteration_records)

                if len(ratio_records) > 0:
                    ratio_df = pd.DataFrame(ratio_records)
                    ratio_df.to_excel(os.path.join(train_fraction_dir, "all_iterations_all_model_results.xlsx"), index=False)
                    ratio_summary, ratio_family_summary = summarize_repeated_results(ratio_df, train_fraction_dir)

    if len(all_records) == 0:
        raise RuntimeError("No valid model was trained.")

    result_df = pd.DataFrame(all_records)
    result_df.to_excel(os.path.join(all_results_dir, "all_repeated_training_model_results.xlsx"), index=False)

    core_cols = [
        "training_iteration", "training_fraction", "test_fraction", "unused_fraction",
        "Model family", "Version", "Weight scheme", "Weight beta", "Weight location",
        "z_score_threshold", "cv", "n_train", "n_test", "n_unused",
        "M0 test R2", "M0 test RMSE", "M0 test MAE",
        "WR0 M2 test R2", "R5C M2 test R2",
        "M2 test R2", "M2 test RMSE", "M2 test MAE",
        "Delta test R2 M2_minus_M0", "Delta test R2 M2_minus_WR0", "Delta test R2 M2_minus_R5C",
        "Delta test RMSE M2_minus_M0", "Delta test RMSE M2_minus_WR0", "Delta test RMSE M2_minus_R5C",
        "M0 overfit gap", "M2 overfit gap",
        "M0_best_params", "Protein_proxy_best_params",
        "Low group n", "High group n", "Residual input feature cols", "Residual model feature cols",
        "Residual train weight_min", "Residual train weight_max", "Residual train weight_mean", "Residual train weight_std",
        "Alpha test weight_min", "Alpha test weight_max", "Alpha test weight_mean", "Alpha test weight_std",
        "Smooth_alpha_base", "Smooth_alpha_amp", "Smooth_alpha_slope", "Smooth_alpha_center_quantile",
        "Smooth_alpha_center", "Smooth_alpha_scale", "Smooth_alpha_direction", "Smooth_alpha_max_clip",
    ]
    core_cols_existing = [c for c in core_cols if c in result_df.columns]
    result_df[core_cols_existing].to_excel(os.path.join(all_results_dir, "all_repeated_training_core_metrics.xlsx"), index=False)

    summary_df, family_summary_df = summarize_repeated_results(result_df, all_results_dir)
    box_dir = plot_repeated_training_boxplots(result_df, screening_root)

    selected_summary, selected_record, selected_bundle, fixed_dir = select_fixed_model_from_repeated_results(
        summary_df,
        best_bundle_by_spec,
        screening_root,
    )
    train_pred_df, test_pred_df = save_fixed_model_predictions(selected_bundle, selected_record, screening_root, original_target_col)
    scatter_dir = plot_fixed_model_scatter(selected_bundle, screening_root)
    mechanism_dir = run_mechanism_validation_outputs(selected_bundle, selected_record, screening_root, base_weather_features)

    # Additional fixed-model diagnostics that are not repeated-training boxplots.
    importance_outputs = plot_importance_outputs(selected_bundle, base_weather_features, fixed_dir)
    metric_df = plot_metric_comparison_for_best(selected_record, fixed_dir)
    weight_df = plot_weight_outputs(selected_record, teacher_clean, selected_bundle["test_df"], fixed_dir)

    # Save a final workbook linking major outputs.
    final_workbook = os.path.join(screening_root, "M3_repeated_training_final_summary.xlsx")
    with pd.ExcelWriter(final_workbook) as writer:
        pd.DataFrame([selected_summary]).to_excel(writer, sheet_name="fixed_group_summary", index=False)
        pd.DataFrame([selected_record]).to_excel(writer, sheet_name="fixed_fitted_record", index=False)
        family_summary_df.to_excel(writer, sheet_name="family_summary", index=False)
        summary_df.head(200).to_excel(writer, sheet_name="top200_grouped_summary", index=False)
        train_pred_df.to_excel(writer, sheet_name="fixed_train_predictions", index=False)
        test_pred_df.to_excel(writer, sheet_name="fixed_validation_predictions", index=False)

    print("\n================ Repeated independent training completed ================")
    print("Screening root:", screening_root)
    print("All results:", all_results_dir)
    print("Repeated-training boxplots:", box_dir)
    print("Fixed model folder:", fixed_dir)
    print("Mechanism validation folder:", mechanism_dir)
    print("Fixed prediction scatter folder:", scatter_dir)
    print("Final summary workbook:", final_workbook)
    print("\nSelected fixed M3 specification:")
    print("Training fraction:", selected_summary["training_fraction"])
    print("Version:", selected_summary["Version"])
    print("Weight scheme:", selected_summary["Weight scheme"])
    print("Weight beta:", selected_summary["Weight beta"])
    print("Weight location:", selected_summary["Weight location"])
    print("Mean validation R2:", selected_summary["M2 test R2 mean"])
    print("Mean validation RMSE:", selected_summary["M2 test RMSE mean"])


if __name__ == "__main__":
    main()
