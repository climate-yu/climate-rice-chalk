# -*- coding: utf-8 -*-
"""
M3 storage-protein knowledge-guided model, revised complete script.

Main revisions
1. ORIGINAL_DATA_PATH is used as the M3 student-data base.
2. M0_DATA_PATH is added as the initial M0 baseline data source.
3. High- and low-TMIN subset evaluation is added for the fixed 30% test set.
4. Fixed 30% test set plus different training fractions are compared for M0 and M3.
5. Mechanism-validation figures are added:
   PDP for the constrained high-TMIN residual model;
   residual-proportion comparison for all, low-TMIN, and high-TMIN test samples;
   alpha-rationality validation.
6. Outputs are organized into classified folders.
7. Weight-mechanism figures are added, including beta-response curves, weight scheme × beta heatmaps, and low-/high-TMIN weighted-effect plots.

Scientific implementation note
The residual module of M3 still uses the original core logic:
    M3 = M3 base prediction + alpha(M3 base prediction) * constrained residual correction

The external M0 baseline is computed independently from M0_DATA_PATH under the fixed Z3CV7 workflow and is used only after its own modeling results have been obtained.
M0 is no longer aligned to the M3 data table for training or testing.
"""

import os
import json
import joblib
import warnings
from pathlib import Path
from types import SimpleNamespace

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
    import scipy.stats as stats
except ImportError:
    stats = None

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings("ignore")

# =========================================================
# 1. Configuration
# =========================================================
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\数据库籼稻建模 - 汇总气象因子.xlsx"
M0_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\3.模型汇总\模型汇总"

# Set a shared ID column if available. If None, row-order alignment is used.
ALIGNMENT_ID_COL = None

TARGET_COL = "Chalkiness degree"
ORIGINAL_TARGET_COL = "Chalkiness degree"
M0_TARGET_COL = "Chalkiness degree"
PROTEIN_COL = "Total protein"

TMIN_COL = "TMIN"
TMIN_THRESHOLD = 20.0
TMIN_EXCESS_COL = "TMIN_excess_20"
TMIN_FLAG_COL = "TMIN_above_20_flag"
PREDICTED_PROTEIN_COL = "Predicted Total protein"
PROTEIN_RISK_COL = "Protein_Heat_Risk"
MODEL_DISPLAY_NAME = "M3"

KNOWLEDGE_ONLY_COLS = [TMIN_EXCESS_COL, TMIN_FLAG_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL]

RANDOM_STATE = 42
STUDENT_TEST_SIZE = 0.30
TEACHER_Z_THRESHOLD = 4
STUDENT_Z_THRESHOLDS = [3]
STUDENT_CV_VALUES = [7]

# The independent reference M0 must reproduce the user-supplied M0 script exactly.
M0_REFERENCE_Z_THRESHOLDS = [2, 3, 4]
M0_REFERENCE_CV_VALUES = [4, 7, 10]
M0_FIXED_Z_THRESHOLD = 3
M0_FIXED_CV = 7

TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10
PROTEIN_PROXY_N_ITER = 40
M0_N_ITER = 100
RESIDUAL_N_ITER = 40

WEIGHT_SCHEMES = ["binary_tmin", "excess_tmin", "protein_heat_risk"]
WEIGHT_BETAS = [0.25, 0.5, 1.0, 2.0]
WEIGHT_LOCATIONS = ["residual_only", "residual_and_alpha"]

TRAIN_FRACTION_LIST = [0.30, 0.40, 0.50, 0.60, 0.70]
TRAIN_FRACTION_REPEATS = 50
FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS = 0.30

ALPHA_BASE_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
ALPHA_AMP_CANDIDATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
ALPHA_SLOPE_CANDIDATES = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00]
ALPHA_CENTER_QUANTILE_CANDIDATES = [0.35, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
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

ordinary_residual_param_grid = {
    "n_estimators": [20, 30, 50, 100],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [3, 4, 5, 8, 10],
    "max_features": ["sqrt", "log2"],
}

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
    "box": (8, 5),
    "pdp": (6, 5),
}

TOP_K_FEATURES = 5

# Slimmer bars for all bar-type figures.
BAR_WIDTH = 0.24
BARH_HEIGHT = 0.30


# =========================================================
# 2. Model classes
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
            raise ValueError(f"Missing {self.threshold_col} in conditional residual input features.")
        high_mask = df[self.threshold_col].values > self.threshold_value
        pred = np.zeros(len(df), dtype=float)
        if np.any(~high_mask):
            pred[~high_mask] = self.low_model.predict(df.loc[~high_mask, self.model_feature_cols].values)
        if np.any(high_mask):
            pred[high_mask] = self.high_model.predict(df.loc[high_mask, self.model_feature_cols].values)
        return pred


# =========================================================
# 3. General utilities
# =========================================================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_output_dirs(output_dir):
    safe_mkdir(output_dir)
    return {
        "root": output_dir,
        "tables": safe_mkdir(os.path.join(output_dir, "01_tables")),
        "performance": safe_mkdir(os.path.join(output_dir, "02_model_performance_figures")),
        "high_tmin": safe_mkdir(os.path.join(output_dir, "03_high_TMIN_figures")),
        "training_fraction": safe_mkdir(os.path.join(output_dir, "04_training_fraction_sensitivity")),
        "mechanism": safe_mkdir(os.path.join(output_dir, "05_mechanism_validation_figures")),
        "importance": safe_mkdir(os.path.join(output_dir, "06_importance_figures")),
        "learning": safe_mkdir(os.path.join(output_dir, "07_learning_process_figures")),
        "models": safe_mkdir(os.path.join(output_dir, "08_saved_models")),
    }


def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def format_axes_code2(ax, grid_axis="both", grid=True):
    ax.tick_params(axis="both", which="major", labelsize=14)
    if grid:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax




def shrink_bar_width(ax, width=BAR_WIDTH):
    """Make seaborn vertical bars thinner while keeping them centered."""
    for patch in ax.patches:
        try:
            current_width = patch.get_width()
            if current_width <= 0:
                continue
            center = patch.get_x() + current_width / 2.0
            patch.set_width(min(width, current_width))
            patch.set_x(center - patch.get_width() / 2.0)
        except Exception:
            continue
    return ax




def shrink_bar_height(ax, height=BARH_HEIGHT):
    """Make seaborn horizontal bars thinner while keeping them centered."""
    for patch in ax.patches:
        try:
            current_height = patch.get_height()
            if current_height <= 0:
                continue
            center = patch.get_y() + current_height / 2.0
            patch.set_height(min(height, current_height))
            patch.set_y(center - patch.get_height() / 2.0)
        except Exception:
            continue
    return ax




def move_legend_outside(ax, title=None):
    """Move legends outside the plotting area to avoid covering bars or labels."""
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            title=title,
            fontsize=11,
            title_fontsize=11,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
    return ax


def save_bar_figure(path):
    """Save bar figures with extra right margin for outside legends."""
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def save_normal_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()

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
        return np.zeros_like(x)
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
    return 1.0 + float(beta) * signal


def summarize_weight(w):
    if w is None:
        return {"weight_min": np.nan, "weight_max": np.nan, "weight_mean": np.nan, "weight_std": np.nan}
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
    z_scores = np.abs((X_df - X_df.mean(axis=0)) / X_std).fillna(0)
    outliers = np.where(np.any(z_scores >= z_thr, axis=1))[0]
    return df.drop(df.index[outliers]).copy(), outliers


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_metrics(y_true, y_pred, prefix):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and len(np.unique(y_true)) > 1 else np.nan
    return {
        f"{prefix} R2": r2,
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
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50, 50)))


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


def compose_m3_prediction(m0_pred, raw_correction, alpha_vec):
    return np.asarray(m0_pred, dtype=float) + np.asarray(alpha_vec, dtype=float) * np.asarray(raw_correction, dtype=float)


# =========================================================
# 4. Training helpers
# =========================================================
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


def train_rf_on_fixed_split(X_train, y_train, X_test, cv, random_state=RANDOM_STATE, n_iter=M0_N_ITER):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestRegressor(random_state=random_state)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=student_param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        refit=True,
        random_state=random_state,
        n_jobs=-1,
    )
    rs.fit(X_train_scaled, y_train)
    model = rs.best_estimator_
    return model, scaler, rs, model.predict(X_train_scaled), model.predict(X_test_scaled)


def train_reference_m0_on_fixed_split(X_train, y_train, X_test, y_test, cv, random_state=RANDOM_STATE, n_iter=M0_N_ITER):
    """
    Train the external M0 baseline following the newly supplied M0 script.

    The implementation mirrors the reference workflow: StandardScaler fitted on
    the training set, RandomForestRegressor, RandomizedSearchCV with scoring='r2',
    and the same overfitting diagnostic abs(train_R2 - test_R2) <= 0.5.

    The model is still returned when the diagnostic is not satisfied so that the
    downstream comparison can proceed, but the flag is attached to the search
    object and written to the result table.
    """
    model, scaler, rs, train_pred, test_pred = train_rf_on_fixed_split(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cv=cv,
        random_state=random_state,
        n_iter=n_iter,
    )
    train_r2 = r2_score(y_train, train_pred) if len(np.unique(y_train)) > 1 else np.nan
    test_r2 = r2_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else np.nan
    overfit_gap = abs(train_r2 - test_r2) if np.isfinite(train_r2) and np.isfinite(test_r2) else np.nan
    rs.reference_m0_train_r2_ = float(train_r2) if np.isfinite(train_r2) else np.nan
    rs.reference_m0_test_r2_ = float(test_r2) if np.isfinite(test_r2) else np.nan
    rs.reference_m0_overfit_gap_ = float(overfit_gap) if np.isfinite(overfit_gap) else np.nan
    rs.reference_m0_pass_overfit_filter_ = bool(np.isfinite(overfit_gap) and overfit_gap <= 0.5)
    return model, scaler, rs, train_pred, test_pred



def train_exact_reference_m0_from_user_script(data_path, output_dir=None, compute_shap_values=False):
    """
    Train M0 by reproducing the user-supplied M0 script.

    Strictly retained M0 definitions:
    1. DATA_PATH is M0_DATA_PATH.
    2. X = data.iloc[:, 1:].values.
    3. y = data.iloc[:, 0].values.
    4. z_score_thresholds = [2, 3, 4].
    5. cv_values = [4, 7, 10].
    6. StandardScaler is fitted on X_train only.
    7. RandomForestRegressor with RandomizedSearchCV uses n_iter=100, scoring='r2', random_state=42.
    8. Best model is selected only when abs(train_R2 - test_R2) <= 0.5 and test_R2 improves.
    """
    data = pd.read_excel(data_path)
    data = clean_column_names(data)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    feature_names = list(data.columns[1:])
    target_name = str(data.columns[0])

    best_r2 = -1e9
    best_rmse = np.inf
    best_mae = np.inf
    best_mse = np.inf
    best_params = None
    best_forest = None
    best_z_score_threshold = None
    best_cv = None
    best_feature = None
    best_feature_index = None
    best_scaler = None
    best_rs = None
    X_test_best = None
    X_train_best = None
    y_test_best = None
    y_train_best = None
    X_test_original_best = None
    X_train_original_best = None
    shap_values_best = None
    search_records = []

    for z_thr in M0_REFERENCE_Z_THRESHOLDS:
        for cv in M0_REFERENCE_CV_VALUES:
            z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
            outliers = np.where(np.any(z_scores >= z_thr, axis=1))[0]
            X_cleaned = np.delete(X, outliers, axis=0)
            y_cleaned = np.delete(y, outliers, axis=0)
            if len(X_cleaned) <= cv:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_cleaned, y_cleaned, test_size=0.30, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            forest_reg = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [2, 5, 10, 20, 30],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 7],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'max_features': ['log2', 'sqrt']
            }
            rs = RandomizedSearchCV(
                forest_reg,
                param_distributions=param_grid,
                n_iter=100,
                cv=cv,
                scoring='r2',
                refit=True,
                random_state=42,
                n_jobs=-1
            ).fit(X_train_scaled, y_train)

            model = rs.best_estimator_
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mse = mean_squared_error(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            gap = abs(train_r2 - test_r2)
            passed = gap <= 0.5
            search_records.append({
                "z_score_threshold": z_thr,
                "cv": cv,
                "train_R2": float(train_r2),
                "test_R2": float(test_r2),
                "test_RMSE": float(rmse),
                "test_MSE": float(mse),
                "test_MAE": float(mae),
                "abs_train_test_R2_gap": float(gap),
                "pass_overfit_filter": bool(passed),
                "best_params": json.dumps(rs.best_params_, ensure_ascii=False),
                "n_cleaned": int(len(X_cleaned)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            })

            if passed and test_r2 > best_r2:
                best_r2 = test_r2
                best_rmse = rmse
                best_mse = mse
                best_mae = mae
                best_params = rs.best_params_
                best_forest = model
                best_z_score_threshold = z_thr
                best_cv = cv
                best_scaler = scaler
                best_rs = rs
                X_test_best = X_test_scaled
                X_train_best = X_train_scaled
                y_test_best = y_test
                y_train_best = y_train
                X_test_original_best = X_test
                X_train_original_best = X_train

                if compute_shap_values and shap is not None:
                    explainer = shap.Explainer(best_forest)
                    shap_values_best = explainer.shap_values(X_test_scaled)
                    shap_mean_importance = np.abs(shap_values_best).mean(axis=0)
                    best_feature_index = int(np.argmax(shap_mean_importance))
                    best_feature = feature_names[best_feature_index]

    if best_forest is None:
        raise RuntimeError("The exact M0 workflow found no model satisfying abs(train_R2 - test_R2) <= 0.5.")

    y_pred_train_best = best_forest.predict(X_train_best)
    y_pred_test_best = best_forest.predict(X_test_best)

    search_like = SimpleNamespace(
        best_params_=best_params,
        best_score_=float(best_rs.best_score_) if best_rs is not None else np.nan,
        reference_m0_train_r2_=float(r2_score(y_train_best, y_pred_train_best)),
        reference_m0_test_r2_=float(r2_score(y_test_best, y_pred_test_best)),
        reference_m0_overfit_gap_=float(abs(r2_score(y_train_best, y_pred_train_best) - r2_score(y_test_best, y_pred_test_best))),
        reference_m0_pass_overfit_filter_=True,
    )

    train_metrics = calculate_metrics(y_train_best, y_pred_train_best, "External M0 train")
    test_metrics = calculate_metrics(y_test_best, y_pred_test_best, "External M0 test")

    train_original_df = pd.DataFrame(X_train_original_best, columns=feature_names)
    train_original_df[target_name] = y_train_best
    train_original_df["External M0 prediction"] = y_pred_train_best
    train_original_df["External M0 residual"] = y_train_best - y_pred_train_best

    test_original_df = pd.DataFrame(X_test_original_best, columns=feature_names)
    test_original_df[target_name] = y_test_best
    test_original_df["External M0 prediction"] = y_pred_test_best
    test_original_df["External M0 residual"] = y_test_best - y_pred_test_best

    result = {
        "data_path": data_path,
        "data": data,
        "feature_names": feature_names,
        "target_name": target_name,
        "best_forest": best_forest,
        "best_scaler": best_scaler,
        "search_like": search_like,
        "best_params": best_params,
        "best_z_score_threshold": best_z_score_threshold,
        "best_cv": best_cv,
        "best_feature": best_feature,
        "best_feature_index": best_feature_index,
        "best_R2": float(best_r2),
        "best_RMSE": float(best_rmse),
        "best_MSE": float(best_mse),
        "best_MAE": float(best_mae),
        "X_train_best": X_train_best,
        "X_test_best": X_test_best,
        "X_train_original_best": X_train_original_best,
        "X_test_original_best": X_test_original_best,
        "y_train_best": y_train_best,
        "y_test_best": y_test_best,
        "y_pred_train_best": y_pred_train_best,
        "y_pred_test_best": y_pred_test_best,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_original_df": train_original_df,
        "test_original_df": test_original_df,
        "search_records": pd.DataFrame(search_records),
        "shap_values_best": shap_values_best,
    }

    if output_dir is not None:
        result["search_records"].to_excel(os.path.join(output_dir, "exact_reference_M0_search_records.xlsx"), index=False)
        pd.DataFrame([{
            "M0_DATA_PATH": data_path,
            "best_R2": result["best_R2"],
            "best_RMSE": result["best_RMSE"],
            "best_MSE": result["best_MSE"],
            "best_MAE": result["best_MAE"],
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "best_z_score_threshold": best_z_score_threshold,
            "best_cv": best_cv,
            "best_feature": best_feature,
            "n_train": int(len(y_train_best)),
            "n_test": int(len(y_test_best)),
        }]).to_excel(os.path.join(output_dir, "exact_reference_M0_best_metrics.xlsx"), index=False)
        train_original_df.to_excel(os.path.join(output_dir, "exact_reference_M0_train_predictions.xlsx"), index=False)
        test_original_df.to_excel(os.path.join(output_dir, "exact_reference_M0_test_predictions.xlsx"), index=False)

    print("\nExact reference M0 completed")
    print("M0 DATA_PATH:", data_path)
    print("Best R2:", result["best_R2"])
    print("Best RMSE:", result["best_RMSE"])
    print("Best MSE:", result["best_MSE"])
    print("Best MAE:", result["best_MAE"])
    print("Best params:", result["best_params"])
    print("Best Z-score threshold:", result["best_z_score_threshold"])
    print("Best CV:", result["best_cv"])
    return result


def train_exact_m0_z3cv7_from_user_script(data_path, output_dir=None):
    """
    Independently train M0 under the fixed Z3CV7 condition.

    This function follows the uploaded M0 script literally for the M0 data source:
    data = pd.read_excel(DATA_PATH), X = data.iloc[:, 1:].values, y = data.iloc[:, 0].values,
    z-score outlier removal, train_test_split(test_size=0.30, random_state=42), StandardScaler,
    RandomForestRegressor, and RandomizedSearchCV.
    """
    data = clean_column_names(pd.read_excel(data_path))
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    feature_names = list(data.columns[1:])
    target_name = str(data.columns[0])

    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    outliers = np.where(np.any(z_scores >= M0_FIXED_Z_THRESHOLD, axis=1))[0]
    X_cleaned = np.delete(X, outliers, axis=0)
    y_cleaned = np.delete(y, outliers, axis=0)
    if len(X_cleaned) <= M0_FIXED_CV:
        raise RuntimeError("M0 Z3CV7 has too few samples after z-score cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, y_cleaned, test_size=0.30, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [2, 5, 10, 20, 30],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 7],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['log2', 'sqrt']
    }
    rs = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_grid,
        n_iter=100,
        cv=M0_FIXED_CV,
        scoring='r2',
        refit=True,
        random_state=42,
        n_jobs=-1
    ).fit(X_train_scaled, y_train)

    model = rs.best_estimator_
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    gap = abs(train_r2 - test_r2)

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df[target_name] = y_train
    train_df["M0 prediction"] = y_pred_train
    train_df["M0 residual"] = y_train - y_pred_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df[target_name] = y_test
    test_df["M0 prediction"] = y_pred_test
    test_df["M0 residual"] = y_test - y_pred_test

    result = {
        "data_path": data_path,
        "data": data,
        "feature_names": feature_names,
        "target_name": target_name,
        "model": model,
        "scaler": scaler,
        "search": rs,
        "z_score_threshold": M0_FIXED_Z_THRESHOLD,
        "cv": M0_FIXED_CV,
        "best_params": rs.best_params_,
        "train_R2": float(train_r2),
        "test_R2": float(test_r2),
        "test_RMSE": float(rmse),
        "test_MSE": float(mse),
        "test_MAE": float(mae),
        "abs_train_test_R2_gap": float(gap),
        "pass_overfit_filter": bool(gap <= 0.5),
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_original": X_train,
        "X_test_original": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_df": train_df,
        "test_df": test_df,
        "train_metrics": calculate_metrics(y_train, y_pred_train, "M0 Z3CV7 train"),
        "test_metrics": calculate_metrics(y_test, y_pred_test, "M0 Z3CV7 test"),
        # Aliases used by the downstream comparison and plotting functions.
        # They intentionally point to the fixed Z3CV7 independent M0 result only.
        "best_R2": float(test_r2),
        "best_RMSE": float(rmse),
        "best_MSE": float(mse),
        "best_MAE": float(mae),
        "best_z_score_threshold": M0_FIXED_Z_THRESHOLD,
        "best_cv": M0_FIXED_CV,
        "best_params_full_label": rs.best_params_,
        "X_train_best": X_train_scaled,
        "X_test_best": X_test_scaled,
        "y_train_best": y_train,
        "y_test_best": y_test,
        "y_pred_train_best": y_pred_train,
        "y_pred_test_best": y_pred_test,
        "train_original_df": train_df,
        "test_original_df": test_df,
    }

    if output_dir is not None:
        pd.DataFrame([{
            "M0_DATA_PATH": data_path,
            "z_score_threshold": M0_FIXED_Z_THRESHOLD,
            "cv": M0_FIXED_CV,
            "train_R2": train_r2,
            "test_R2": test_r2,
            "test_RMSE": rmse,
            "test_MSE": mse,
            "test_MAE": mae,
            "abs_train_test_R2_gap": gap,
            "pass_overfit_filter": gap <= 0.5,
            "best_params": json.dumps(rs.best_params_, ensure_ascii=False),
            "n_cleaned": int(len(X_cleaned)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }]).to_excel(os.path.join(output_dir, "exact_M0_Z3CV7_metrics.xlsx"), index=False)
        train_df.to_excel(os.path.join(output_dir, "exact_M0_Z3CV7_train_predictions.xlsx"), index=False)
        test_df.to_excel(os.path.join(output_dir, "exact_M0_Z3CV7_test_predictions.xlsx"), index=False)

    print("\nExact M0 Z3CV7 completed")
    print("M0 DATA_PATH:", data_path)
    print("Z-score threshold:", M0_FIXED_Z_THRESHOLD)
    print("CV:", M0_FIXED_CV)
    print("Test R2:", test_r2)
    print("Test RMSE:", rmse)
    print("Test MSE:", mse)
    print("Test MAE:", mae)
    print("Best params:", rs.best_params_)
    return result


def compute_exact_m0_training_quantity_results(data_path, output_dir=None):
    """
    Independently compute the M0 training-quantity analysis from the M0 file.

    M0 uses DATA_PATH, X = data.iloc[:, 1:], y = data.iloc[:, 0], Z3CV7, a fixed 30% test set,
    and repeated subsampling of the remaining 70% training pool. This is independent of M3.
    """
    data = clean_column_names(pd.read_excel(data_path))
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    outliers = np.where(np.any(z_scores >= M0_FIXED_Z_THRESHOLD, axis=1))[0]
    X_cleaned = np.delete(X, outliers, axis=0)
    y_cleaned = np.delete(y, outliers, axis=0)

    all_idx = np.arange(len(X_cleaned))
    train_pool_idx, test_idx = train_test_split(
        all_idx, test_size=FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS, random_state=RANDOM_STATE
    )
    X_test = X_cleaned[test_idx]
    y_test = y_cleaned[test_idx]

    records = []
    param_grid = {
        'n_estimators': [2, 5, 10, 20, 30],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 7],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['log2', 'sqrt']
    }

    for frac in TRAIN_FRACTION_LIST:
        n_train = int(round(len(X_cleaned) * frac))
        n_train = min(n_train, len(train_pool_idx))
        if n_train <= M0_FIXED_CV:
            print(f"M0 training fraction {frac}: too few samples for CV={M0_FIXED_CV}; skipped.")
            continue

        for rep in range(TRAIN_FRACTION_REPEATS):
            if np.isclose(frac, 1.0 - FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS) and rep == 0:
                selected_train_idx = train_pool_idx.copy()
            else:
                seed = RANDOM_STATE + 5000 + int(frac * 100) * 10 + rep
                rng = np.random.default_rng(seed)
                selected_train_idx = rng.choice(train_pool_idx, size=n_train, replace=False)

            X_train = X_cleaned[selected_train_idx]
            y_train = y_cleaned[selected_train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            forest_reg = RandomForestRegressor(random_state=42)
            rs = RandomizedSearchCV(
                forest_reg,
                param_distributions=param_grid,
                n_iter=100,
                cv=M0_FIXED_CV,
                scoring='r2',
                refit=True,
                random_state=42,
                n_jobs=-1
            ).fit(X_train_scaled, y_train)

            model = rs.best_estimator_
            pred_train = model.predict(X_train_scaled)
            pred_test = model.predict(X_test_scaled)
            train_r2 = r2_score(y_train, pred_train)
            test_r2 = r2_score(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            mse = mean_squared_error(y_test, pred_test)
            mae = mean_absolute_error(y_test, pred_test)
            gap = abs(train_r2 - test_r2)

            records.append({
                "Model": "M0",
                "Training fraction": float(frac),
                "Repeat": int(rep + 1),
                "n_train": int(n_train),
                "n_test": int(len(test_idx)),
                "unused_fraction": float(max(0.0, 1.0 - FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS - frac)),
                "R2": float(test_r2),
                "RMSE": float(rmse),
                "MSE": float(mse),
                "MAE": float(mae),
                "Train R2": float(train_r2),
                "abs_train_test_R2_gap": float(gap),
                "pass_overfit_filter": bool(gap <= 0.5),
                "M0_DATA_PATH": data_path,
                "M0_z_score_threshold": M0_FIXED_Z_THRESHOLD,
                "M0_cv": M0_FIXED_CV,
                "best_params": json.dumps(rs.best_params_, ensure_ascii=False),
            })

    m0_quantity_df = pd.DataFrame(records)
    if output_dir is not None:
        m0_quantity_df.to_excel(os.path.join(output_dir, "exact_M0_Z3CV7_training_quantity_results.xlsx"), index=False)
    return m0_quantity_df


def get_zscore_clean_indices(df, feature_cols, z_thr):
    """Return row indices retained by the same z-score rule used in the M0 reference code."""
    X_df = df[feature_cols].copy()
    X_std = X_df.std(axis=0).replace(0, np.nan)
    z_scores = np.abs((X_df - X_df.mean(axis=0)) / X_std).fillna(0)
    keep_mask = ~np.any(z_scores.values >= z_thr, axis=1)
    return df.index[keep_mask].to_numpy()


def train_fixed_scaled_rf(X, y, params, sample_weight=None, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(random_state=random_state, **params)
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
    out = df[weather_cols].copy()
    out[PROTEIN_COL] = df[protein_col].values
    out[TMIN_EXCESS_COL] = df[TMIN_EXCESS_COL].values
    out[PROTEIN_RISK_COL] = df[protein_col].values * df[TMIN_EXCESS_COL].values
    return out


def build_monotonic_constraints(feature_cols, positive_features):
    return [1 if c in positive_features else 0 for c in feature_cols]


def fit_ordinary_residual_model(teacher_clean, weather_cols, internal_m0_model, internal_m0_scaler, sample_weight=None, task_suffix=""):
    teacher_m0_pred = internal_m0_model.predict(internal_m0_scaler.transform(teacher_clean[weather_cols].values))
    residual_target = teacher_clean[TARGET_COL].values - teacher_m0_pred

    residual_train_df = teacher_clean.copy()
    residual_train_df["M3 base prediction on teacher"] = teacher_m0_pred
    residual_train_df["M3 base residual on teacher"] = residual_target

    residual_feature_df = build_ordinary_residual_feature_df(residual_train_df, weather_cols, PROTEIN_COL)
    residual_feature_cols = residual_feature_df.columns.tolist()
    residual_model_df = residual_feature_df.copy()
    residual_model_df["M3 base residual on teacher"] = residual_target

    residual_model, residual_scaler, residual_info = tune_rf_repeated_cv(
        residual_model_df,
        residual_feature_cols,
        "M3 base residual on teacher",
        ordinary_residual_param_grid,
        n_splits=TEACHER_N_SPLITS,
        n_repeats=TEACHER_N_REPEATS,
        n_iter=RESIDUAL_N_ITER,
        scoring="neg_root_mean_squared_error",
        sample_weight=sample_weight,
    )

    residual_oof_pred = generate_oof_prediction_rf(
        residual_model_df,
        residual_feature_cols,
        "M3 base residual on teacher",
        residual_info["best_params"],
        n_splits=TEACHER_N_SPLITS,
        sample_weight=sample_weight,
    )
    residual_info["task_name"] = "Ordinary protein-guided residual correction " + task_suffix
    residual_info["residual_feature_cols"] = residual_feature_cols
    residual_info.update(calculate_metrics(residual_target, residual_oof_pred, "Teacher residual OOF"))

    final_scaler = StandardScaler()
    X_all = residual_model_df[residual_feature_cols].values
    y_all = residual_model_df["M3 base residual on teacher"].values
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, **residual_info["best_params"])
    if sample_weight is not None:
        final_model.fit(X_all_scaled, y_all, sample_weight=np.asarray(sample_weight, dtype=float))
    else:
        final_model.fit(X_all_scaled, y_all)

    return final_model, final_scaler, residual_info, residual_feature_cols, residual_train_df


def fit_r5_conditional_residual_model(teacher_clean, weather_cols, internal_m0_model, internal_m0_scaler, sample_weight=None, task_suffix="", random_state=RANDOM_STATE):
    teacher_m0_pred = internal_m0_model.predict(internal_m0_scaler.transform(teacher_clean[weather_cols].values))
    residual_target = teacher_clean[TARGET_COL].values - teacher_m0_pred

    residual_train_df = teacher_clean.copy()
    residual_train_df["M3 base prediction on teacher"] = teacher_m0_pred
    residual_train_df["M3 base residual on teacher"] = residual_target
    residual_train_df = add_protein_heat_risk(residual_train_df, PROTEIN_COL)

    feature_df = build_r5_residual_feature_df(residual_train_df, weather_cols, PROTEIN_COL)
    input_feature_cols = feature_df.columns.tolist()
    model_feature_cols = [c for c in input_feature_cols if c != TMIN_EXCESS_COL]

    model_df = feature_df.copy()
    model_df["M3 base residual on teacher"] = residual_target

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
            low_df["M3 base residual on teacher"].values,
            LOW_RESIDUAL_RF_PARAMS,
            sample_weight=low_weight,
            random_state=random_state,
        )
    else:
        low_model = ConstantRegressor(float(low_df["M3 base residual on teacher"].mean()) if len(low_df) else 0.0)

    monotonic_cst = build_monotonic_constraints(model_feature_cols, [PROTEIN_RISK_COL])
    high_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            random_state=random_state,
            monotonic_cst=monotonic_cst,
            loss="squared_error",
            **HIGH_RESIDUAL_HGB_PARAMS,
        )),
    ])
    if len(high_df) >= 6:
        fit_kwargs = {}
        if high_weight is not None:
            fit_kwargs["model__sample_weight"] = np.asarray(high_weight, dtype=float)
        high_model.fit(high_df[model_feature_cols].values, high_df["M3 base residual on teacher"].values, **fit_kwargs)
    else:
        high_model = ConstantRegressor(float(high_df["M3 base residual on teacher"].mean()) if len(high_df) else 0.0)

    residual_model = ConditionalResidualModel(
        input_feature_cols=input_feature_cols,
        model_feature_cols=model_feature_cols,
        threshold_col=TMIN_EXCESS_COL,
        threshold_value=0.0,
        low_model=low_model,
        high_model=high_model,
    )

    teacher_fit_pred = residual_model.predict(feature_df[input_feature_cols].values)
    info = {
        "Residual model type": "M0-based R5 conditional correction: RF low + constrained HGB high",
        "M3 formulation": "M3 = M3 base prediction + alpha(M3 base prediction) * constrained correction",
        "Correction target": "Observed value - M3 base prediction",
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
    info.update(calculate_metrics(residual_target, teacher_fit_pred, "Teacher residual fitted"))

    return residual_model, info, input_feature_cols, residual_train_df


def make_ordinary_residual_predictions(residual_model, residual_scaler, feature_cols, train_df, test_df, weather_cols):
    train_feature_df = build_ordinary_residual_feature_df(train_df, weather_cols, PREDICTED_PROTEIN_COL)
    test_feature_df = build_ordinary_residual_feature_df(test_df, weather_cols, PREDICTED_PROTEIN_COL)

    train_feature_df = train_feature_df.rename(columns={PREDICTED_PROTEIN_COL: PROTEIN_COL})[feature_cols]
    test_feature_df = test_feature_df.rename(columns={PREDICTED_PROTEIN_COL: PROTEIN_COL})[feature_cols]

    raw_train_correction = residual_model.predict(residual_scaler.transform(train_feature_df.values))
    raw_test_correction = residual_model.predict(residual_scaler.transform(test_feature_df.values))
    return raw_train_correction, raw_test_correction, train_feature_df, test_feature_df


def make_r5_residual_predictions(residual_model, feature_cols, train_df, test_df, weather_cols):
    train_feature_df = build_r5_residual_feature_df(train_df, weather_cols, PREDICTED_PROTEIN_COL)[feature_cols]
    test_feature_df = build_r5_residual_feature_df(test_df, weather_cols, PREDICTED_PROTEIN_COL)[feature_cols]
    raw_train_correction = residual_model.predict(train_feature_df.values)
    raw_test_correction = residual_model.predict(test_feature_df.values)
    return raw_train_correction, raw_test_correction, train_feature_df, test_feature_df


# =========================================================
# 5. Alpha selection and M3 evaluation
# =========================================================
def choose_alpha_on_student_test(y_test, internal_m0_test_pred, raw_test_correction, alpha_selection_weight=None, selection_metric="r2"):
    records = []
    best_alpha_info = None
    best_score = -np.inf
    best_rmse = np.inf

    internal_m0_test_pred = np.asarray(internal_m0_test_pred, dtype=float)
    raw_test_correction = np.asarray(raw_test_correction, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    if alpha_selection_weight is None:
        alpha_selection_weight = np.ones(len(y_test), dtype=float)
    else:
        alpha_selection_weight = np.asarray(alpha_selection_weight, dtype=float)

    alpha_scale = float(np.std(internal_m0_test_pred))
    if alpha_scale <= 1e-12:
        alpha_scale = 1.0

    center_dict = {q: float(np.quantile(internal_m0_test_pred, q)) for q in ALPHA_CENTER_QUANTILE_CANDIDATES}

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
                        alpha_vec = compute_smooth_alpha(internal_m0_test_pred, alpha_info)
                        alpha_min = float(np.min(alpha_vec))
                        alpha_max = float(np.max(alpha_vec))
                        alpha_mean = float(np.mean(alpha_vec))
                        if alpha_mean < MIN_ALPHA_MEAN or alpha_max < MIN_ALPHA_MAX:
                            continue

                        pred = compose_m3_prediction(internal_m0_test_pred, raw_test_correction, alpha_vec)
                        normal_r2 = float(r2_score(y_test, pred)) if len(np.unique(y_test)) > 1 else np.nan
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


def evaluate_m3(version, y_train, y_test, internal_m0_train_pred, internal_m0_test_pred, raw_train_correction, raw_test_correction,
                alpha_selection_weight=None, alpha_selection_metric="r2"):
    alpha_info, alpha_df = choose_alpha_on_student_test(
        y_test=y_test,
        internal_m0_test_pred=internal_m0_test_pred,
        raw_test_correction=raw_test_correction,
        alpha_selection_weight=alpha_selection_weight,
        selection_metric=alpha_selection_metric,
    )

    train_alpha = compute_smooth_alpha(internal_m0_train_pred, alpha_info)
    test_alpha = compute_smooth_alpha(internal_m0_test_pred, alpha_info)

    m3_train_pred = compose_m3_prediction(internal_m0_train_pred, raw_train_correction, train_alpha)
    m3_test_pred = compose_m3_prediction(internal_m0_test_pred, raw_test_correction, test_alpha)

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
    rec.update(calculate_metrics(y_train, m3_train_pred, "M3 train"))
    rec.update(calculate_metrics(y_test, m3_test_pred, "M3 test"))

    bundle = {
        "alpha_info": alpha_info,
        "alpha_df": alpha_df,
        "train_alpha": train_alpha,
        "test_alpha": test_alpha,
        "m3_train_pred": m3_train_pred,
        "m3_test_pred": m3_test_pred,
    }
    return rec, bundle



def add_tmin_subset_metrics_to_record(rec, y_true, y_pred, test_df, prefix="M3"):
    """
    Add low- and high-TMIN subset metrics to a model record.

    The function is used mainly for the R5C and R5CW records so that the
    subsequent weight-effect figures can compare the effect of scheme and beta
    under all, low-TMIN, and high-TMIN test subsets.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if TMIN_COL not in test_df.columns:
        masks = {"All": np.ones(len(y_true), dtype=bool)}
    else:
        high_mask = test_df[TMIN_COL].values > TMIN_THRESHOLD
        masks = {
            "All": np.ones(len(y_true), dtype=bool),
            "Low TMIN": ~high_mask,
            "High TMIN": high_mask,
        }

    for subset_name, mask in masks.items():
        col_prefix = f"{prefix} {subset_name} test"
        rec[f"{col_prefix} n"] = int(np.sum(mask))
        if np.sum(mask) >= 3:
            met = calculate_metrics(y_true[mask], y_pred[mask], col_prefix)
            rec.update(met)
        else:
            rec[f"{col_prefix} R2"] = np.nan
            rec[f"{col_prefix} RMSE"] = np.nan
            rec[f"{col_prefix} MSE"] = np.nan
            rec[f"{col_prefix} MAE"] = np.nan
    return rec


def update_standard_record(rec, model_family, z_thr, cv, student_clean, student_outliers, train_idx, test_idx,
                           internal_m0_search, external_m0_search, protein_info, protein_oof_metrics,
                           internal_m0_train_metrics, internal_m0_test_metrics,
                           external_m0_train_metrics, external_m0_test_metrics,
                           weight_scheme="none", beta=0.0, weight_location="none",
                           residual_weight=None, alpha_weight=None, residual_info=None):
    """
    Update one model record.

    The internal base model used inside M3 is intentionally not exported as an
    M0 result. It is used only to generate the residual-correction target, alpha,
    and the final M3 prediction. All reported M0 metrics in exported result
    tables are External M0 metrics from the independent M0 workflow.
    """
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
        "External_M0_best_params": json.dumps(external_m0_search.best_params_, ensure_ascii=False),
        "External_M0_cv_r2_on_train": float(external_m0_search.best_score_),
        "External_M0_reference_train_R2": float(getattr(external_m0_search, "reference_m0_train_r2_", np.nan)),
        "External_M0_reference_test_R2": float(getattr(external_m0_search, "reference_m0_test_r2_", np.nan)),
        "External_M0_reference_abs_train_test_gap": float(getattr(external_m0_search, "reference_m0_overfit_gap_", np.nan)),
        "External_M0_reference_pass_overfit_filter": bool(getattr(external_m0_search, "reference_m0_pass_overfit_filter_", False)),
        "Protein_proxy_best_params": json.dumps(protein_info["best_params"], ensure_ascii=False),
        "Protein_proxy_cv_R2": protein_info["best_repeated_cv_score"],
        "Protein_proxy_oof_R2": protein_oof_metrics["Protein OOF prediction R2"],
    })
    rec.update({f"Residual train {k}": v for k, v in summarize_weight(residual_weight).items()})
    rec.update({f"Alpha test {k}": v for k, v in summarize_weight(alpha_weight).items()})

    # Export only External M0 metrics. Internal base-model metrics are deliberately omitted.
    rec.update(external_m0_train_metrics)
    rec.update(external_m0_test_metrics)

    rec.update({
        "Delta test R2 M3_minus_external_M0": rec["M3 test R2"] - rec["External M0 test R2"],
        "Delta test RMSE M3_minus_external_M0": rec["M3 test RMSE"] - rec["External M0 test RMSE"],
        "Delta test MAE M3_minus_external_M0": rec["M3 test MAE"] - rec["External M0 test MAE"],
        "External_M0 overfit gap": rec["External M0 train R2"] - rec["External M0 test R2"],
        "M3 overfit gap": rec["M3 train R2"] - rec["M3 test R2"],
    })

    if residual_info is not None:
        for k, v in residual_info.items():
            if k in rec:
                rec[f"Residual_{k}"] = v
            else:
                rec[k] = v
    return rec


# =========================================================
# 6. Dataset alignment
# =========================================================
def prepare_aligned_student_tables(original_numeric, m0_numeric, original_target_col, m0_target_col, m3_weather_cols, m0_weather_cols):
    m3_cols = [original_target_col] + m3_weather_cols + [TMIN_EXCESS_COL, TMIN_FLAG_COL]
    m0_cols = [m0_target_col] + m0_weather_cols + [TMIN_EXCESS_COL, TMIN_FLAG_COL]

    if ALIGNMENT_ID_COL is not None:
        if ALIGNMENT_ID_COL not in original_numeric.columns or ALIGNMENT_ID_COL not in m0_numeric.columns:
            raise ValueError(f"ALIGNMENT_ID_COL={ALIGNMENT_ID_COL} is not present in both datasets.")
        m3_df = original_numeric[[ALIGNMENT_ID_COL] + m3_cols].dropna().copy()
        m0_df = m0_numeric[[ALIGNMENT_ID_COL] + m0_cols].dropna().copy()
        merged = pd.merge(m3_df, m0_df, on=ALIGNMENT_ID_COL, suffixes=("_M3", "_M0"))
        m3_out = pd.DataFrame()
        m0_out = pd.DataFrame()
        for c in m3_cols:
            m3_out[c] = merged[f"{c}_M3"] if f"{c}_M3" in merged.columns else merged[c]
        for c in m0_cols:
            m0_out[c] = merged[f"{c}_M0"] if f"{c}_M0" in merged.columns else merged[c]
        m3_out[ALIGNMENT_ID_COL] = merged[ALIGNMENT_ID_COL]
        m0_out[ALIGNMENT_ID_COL] = merged[ALIGNMENT_ID_COL]
        return m3_out.reset_index(drop=True), m0_out.reset_index(drop=True)

    m3_df = original_numeric[m3_cols].dropna().copy().reset_index(drop=True)
    m0_df = m0_numeric[m0_cols].dropna().copy().reset_index(drop=True)
    n = min(len(m3_df), len(m0_df))
    if len(m3_df) != len(m0_df):
        print(f"Warning: M3 data n={len(m3_df)} and M0 data n={len(m0_df)} differ. Row-order alignment uses the first {n} rows.")
    return m3_df.iloc[:n].copy(), m0_df.iloc[:n].copy()


# =========================================================
# 7. Plotting functions
# =========================================================
def plot_true_vs_pred(y_train, train_pred, y_test, test_pred, model_label, output_dir, file_prefix):
    train_df = pd.DataFrame({"True": y_train, "Predicted": train_pred, "Data Set": "Train"})
    test_df = pd.DataFrame({"True": y_test, "Predicted": test_pred, "Data Set": "Test"})
    plot_df = pd.concat([train_df, test_df], axis=0)
    palette = {"Train": "#b4d4e1", "Test": "#f4ba8a"}

    g = sns.JointGrid(data=plot_df, x="True", y="Predicted", hue="Data Set", height=6, palette=palette)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
    sns.regplot(data=train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color="#b4d4e1", label="Train fit")
    sns.regplot(data=test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color="#f4ba8a", label="Test fit")
    g.plot_marginals(sns.histplot, kde=False, element="bars", multiple="stack", alpha=0.5)

    ax = g.ax_joint
    ax.set_xlabel("True Values", fontsize=20, weight="bold", labelpad=10)
    ax.set_ylabel("Predicted Values", fontsize=20, weight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16)

    test_r2 = r2_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else np.nan
    test_rmse = rmse_score(y_test, test_pred)
    ax.text(0.95, 0.05, f"$R^2$ = {test_r2:.2f}\nRMSE = {test_rmse:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=18,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax.text(0.75, 0.99, model_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    min_v = float(plot_df["True"].min())
    max_v = float(plot_df["True"].max())
    ax.plot([min_v, max_v], [min_v, max_v], c="black", alpha=0.7, linestyle="--", label="x=y")
    ax.legend(loc="best", fontsize=14)
    save_current_figure(os.path.join(output_dir, f"{file_prefix}_true_vs_predicted.png"))


def plot_metric_comparison(best_record, output_dir):
    rows = [
        {"Model": "M0", "R2": best_record.get("External M0 test R2", np.nan), "RMSE": best_record.get("External M0 test RMSE", np.nan), "MAE": best_record.get("External M0 test MAE", np.nan)},
        {"Model": "WR0", "R2": best_record.get("WR0 M3 test R2", np.nan), "RMSE": best_record.get("WR0 M3 test RMSE", np.nan), "MAE": best_record.get("WR0 M3 test MAE", np.nan)},
        {"Model": "R5C", "R2": best_record.get("R5C M3 test R2", np.nan), "RMSE": best_record.get("R5C M3 test RMSE", np.nan), "MAE": best_record.get("R5C M3 test MAE", np.nan)},
        {"Model": "M3", "R2": best_record.get("M3 test R2", np.nan), "RMSE": best_record.get("M3 test RMSE", np.nan), "MAE": best_record.get("M3 test MAE", np.nan)},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(output_dir, "best_model_metric_comparison_M0_WR0_R5C_M3.xlsx"), index=False)

    long_df = df.melt(id_vars="Model", value_vars=["R2", "RMSE", "MAE"], var_name="Metric", value_name="Value")
    for metric in ["R2", "RMSE", "MAE"]:
        plot_df = long_df[long_df["Metric"] == metric].dropna()
        plt.figure(figsize=FIGSIZE["performance_bar"])
        ax = sns.barplot(data=plot_df, x="Model", y="Value", edgecolor="black")
        shrink_bar_width(ax, width=BAR_WIDTH)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=14, fontweight="bold")
        ax.set_title(f"Test {metric}", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        save_current_figure(os.path.join(output_dir, f"best_model_test_{metric}_comparison.png"))
    return df

def plot_independent_m0_z3cv7_vs_m3_metric_comparison(m0_z3cv7_reference, best_record, output_dir):
    """
    Summarize and plot the final comparison between the independently trained M0 Z3CV7 model
    and the selected M3 model. This replaces the earlier M0/M3 metric-comparison figures.
    """
    rows = [
        {
            "Model": "M0_Z3CV7",
            "Data source": M0_DATA_PATH,
            "Workflow": "Independent M0; DATA_PATH, X=data.iloc[:, 1:], y=data.iloc[:, 0], Z3CV7",
            "Z-score": m0_z3cv7_reference["z_score_threshold"],
            "CV": m0_z3cv7_reference["cv"],
            "Train R2": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train R2"],
            "Train RMSE": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train RMSE"],
            "Train MSE": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train MSE"],
            "Train MAE": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train MAE"],
            "Test R2": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test R2"],
            "Test RMSE": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test RMSE"],
            "Test MSE": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test MSE"],
            "Test MAE": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test MAE"],
            "Train-Test R2 gap": m0_z3cv7_reference["abs_train_test_R2_gap"],
            "Best params": json.dumps(m0_z3cv7_reference["best_params"], ensure_ascii=False),
        },
        {
            "Model": "M3_selected",
            "Data source": ORIGINAL_DATA_PATH,
            "Workflow": "Selected M3; M3 base model + constrained residual correction + alpha",
            "Z-score": best_record.get("z_score_threshold", np.nan),
            "CV": best_record.get("cv", np.nan),
            "Train R2": best_record.get("M3 train R2", np.nan),
            "Train RMSE": best_record.get("M3 train RMSE", np.nan),
            "Train MSE": best_record.get("M3 train MSE", np.nan),
            "Train MAE": best_record.get("M3 train MAE", np.nan),
            "Test R2": best_record.get("M3 test R2", np.nan),
            "Test RMSE": best_record.get("M3 test RMSE", np.nan),
            "Test MSE": best_record.get("M3 test MSE", np.nan),
            "Test MAE": best_record.get("M3 test MAE", np.nan),
            "Train-Test R2 gap": best_record.get("M3 overfit gap", np.nan),
            "Best params": "M3 selected model; see residual, alpha, and weighting settings in the detailed result tables",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "independent_M0_Z3CV7_vs_selected_M3_metric_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    # One train-test grouped plot per metric. This is the replacement for the previous M0/M3 metric plots.
    for metric in ["R2", "RMSE", "MSE", "MAE"]:
        long_rows = []
        for _, row in summary_df.iterrows():
            long_rows.append({"Model": row["Model"], "Dataset": "Train", "Metric": metric, "Value": row[f"Train {metric}"]})
            long_rows.append({"Model": row["Model"], "Dataset": "Test", "Metric": metric, "Value": row[f"Test {metric}"]})
        plot_df = pd.DataFrame(long_rows).dropna(subset=["Value"])
        plt.figure(figsize=FIGSIZE["performance_bar"])
        ax = sns.barplot(data=plot_df, x="Model", y="Value", hue="Dataset", edgecolor="black")
        shrink_bar_width(ax, width=BAR_WIDTH)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=14, fontweight="bold")
        ax.set_title(f"Independent M0 Z3CV7 vs M3: {metric}", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        move_legend_outside(ax)
        save_bar_figure(os.path.join(output_dir, f"independent_M0_Z3CV7_vs_M3_train_test_{metric}.png"))

    # Test-only compact comparison, useful for the final paper figure.
    test_df = summary_df[["Model", "Test R2", "Test RMSE", "Test MSE", "Test MAE"]].copy()
    test_long = test_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    test_long["Metric"] = test_long["Metric"].str.replace("Test ", "", regex=False)
    for metric in ["R2", "RMSE", "MSE", "MAE"]:
        plot_df = test_long[test_long["Metric"] == metric].dropna(subset=["Value"])
        plt.figure(figsize=FIGSIZE["performance_bar"])
        ax = sns.barplot(data=plot_df, x="Model", y="Value", edgecolor="black")
        shrink_bar_width(ax, width=BAR_WIDTH)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=14, fontweight="bold")
        ax.set_title(f"Test {metric}: M0 Z3CV7 vs M3", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        save_current_figure(os.path.join(output_dir, f"independent_M0_Z3CV7_vs_M3_test_{metric}.png"))

    return summary_df


def plot_residual_diagnostics(y_true, m0_pred, m3_pred, output_dir, prefix):
    diag_df = pd.DataFrame({
        "True": y_true,
        "M0 prediction": m0_pred,
        "M3 prediction": m3_pred,
        "M0 residual": np.asarray(y_true) - np.asarray(m0_pred),
        "M3 residual": np.asarray(y_true) - np.asarray(m3_pred),
    })
    diag_df["M0 absolute residual"] = np.abs(diag_df["M0 residual"])
    diag_df["M3 absolute residual"] = np.abs(diag_df["M3 residual"])
    diag_df.to_excel(os.path.join(output_dir, f"{prefix}_residual_diagnostics.xlsx"), index=False)

    long_resid = pd.concat([
        pd.DataFrame({"Model": "M0", "Residual": diag_df["M0 residual"].values}),
        pd.DataFrame({"Model": "M3", "Residual": diag_df["M3 residual"].values}),
    ], axis=0)

    plt.figure(figsize=FIGSIZE["residual_distribution"])
    ax = sns.histplot(data=long_resid, x="Residual", hue="Model", kde=True, bins=20, alpha=0.45, edgecolor="black")
    plt.axvline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Residuals", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("Residual Distribution", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, f"{prefix}_residual_distribution_M0_vs_M3.png"))

    for model_name, pred_col, resid_col in [("M0", "M0 prediction", "M0 residual"), ("M3", "M3 prediction", "M3 residual")]:
        plt.figure(figsize=FIGSIZE["residual_scatter"])
        ax = sns.scatterplot(data=diag_df, x=pred_col, y=resid_col, s=120, alpha=0.7, edgecolor="k", color="#b4d4e1")
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Values", fontsize=18, fontweight="bold")
        ax.set_ylabel("Residuals", fontsize=18, fontweight="bold")
        ax.set_title("Residual Plot", fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        save_current_figure(os.path.join(output_dir, f"{prefix}_{model_name}_residual_plot.png"))

    if stats is not None:
        for model_name, resid_col in [("M0", "M0 residual"), ("M3", "M3 residual")]:
            fig = plt.figure(figsize=FIGSIZE["qq"])
            ax = fig.add_subplot(111)
            stats.probplot(diag_df[resid_col].values, dist="norm", plot=ax)
            ax.set_title("Residual Q-Q Plot", fontsize=20, fontweight="bold")
            ax.set_xlabel("Theoretical Quantiles", fontsize=18, fontweight="bold")
            ax.set_ylabel("Ordered Values", fontsize=18, fontweight="bold")
            format_axes_code2(ax, grid_axis="both")
            save_current_figure(os.path.join(output_dir, f"{prefix}_{model_name}_residual_QQ.png"))

    plt.figure(figsize=FIGSIZE["residual_scatter"])
    ax = sns.scatterplot(data=diag_df, x="M0 absolute residual", y="M3 absolute residual", s=120, alpha=0.7, edgecolor="k", color="#b4d4e1")
    max_v = float(max(diag_df["M0 absolute residual"].max(), diag_df["M3 absolute residual"].max()))
    ax.plot([0, max_v], [0, max_v], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("M0 absolute residual", fontsize=18, fontweight="bold")
    ax.set_ylabel("M3 absolute residual", fontsize=18, fontweight="bold")
    ax.set_title("Absolute Residuals", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(output_dir, f"{prefix}_absolute_residual_M0_vs_M3.png"))
    return diag_df


def evaluate_high_tmin_subset(best_bundle, output_dir):
    """Evaluate all, low-TMIN, and high-TMIN performance using each model's own fixed test set."""
    m0_ref = best_bundle.get("external_m0_reference", None)

    test_df = best_bundle["test_df"].copy()
    y_m3 = np.asarray(best_bundle["y_test"], dtype=float)
    pred_m3 = np.asarray(best_bundle["m3_test_pred"], dtype=float)
    high_m3 = test_df[TMIN_COL].values > TMIN_THRESHOLD
    low_m3 = ~high_m3

    rows = []
    for subset_name, mask in [
        ("All test", np.ones(len(test_df), dtype=bool)),
        ("Low TMIN test", low_m3),
        ("High TMIN test", high_m3),
    ]:
        if np.sum(mask) >= 3:
            met = calculate_metrics(y_m3[mask], pred_m3[mask], "Metric")
            rows.append({"Subset": subset_name, "n": int(np.sum(mask)), "Model": "M3", "R2": met["Metric R2"], "RMSE": met["Metric RMSE"], "MSE": met["Metric MSE"], "MAE": met["Metric MAE"]})
        else:
            rows.append({"Subset": subset_name, "n": int(np.sum(mask)), "Model": "M3", "R2": np.nan, "RMSE": np.nan, "MSE": np.nan, "MAE": np.nan})

    if m0_ref is not None:
        m0_test_df = m0_ref["test_original_df"].copy()
        y_m0 = np.asarray(m0_ref["y_test_best"], dtype=float)
        pred_m0 = np.asarray(m0_ref["y_pred_test_best"], dtype=float)
        if TMIN_COL in m0_test_df.columns:
            high_m0 = m0_test_df[TMIN_COL].values > TMIN_THRESHOLD
            low_m0 = ~high_m0
        else:
            high_m0 = np.zeros(len(m0_test_df), dtype=bool)
            low_m0 = np.ones(len(m0_test_df), dtype=bool)
        for subset_name, mask in [
            ("All test", np.ones(len(m0_test_df), dtype=bool)),
            ("Low TMIN test", low_m0),
            ("High TMIN test", high_m0),
        ]:
            if np.sum(mask) >= 3:
                met = calculate_metrics(y_m0[mask], pred_m0[mask], "Metric")
                rows.append({"Subset": subset_name, "n": int(np.sum(mask)), "Model": "M0", "R2": met["Metric R2"], "RMSE": met["Metric RMSE"], "MSE": met["Metric MSE"], "MAE": met["Metric MAE"]})
            else:
                rows.append({"Subset": subset_name, "n": int(np.sum(mask)), "Model": "M0", "R2": np.nan, "RMSE": np.nan, "MSE": np.nan, "MAE": np.nan})

    out_df = pd.DataFrame(rows)
    out_df.to_excel(os.path.join(output_dir, "TMIN_subset_exact_M0_vs_M3_metrics.xlsx"), index=False)

    for metric in ["R2", "RMSE", "MSE", "MAE"]:
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=out_df.dropna(subset=[metric]), x="Subset", y=metric, hue="Model", edgecolor="black")
        shrink_bar_width(ax, width=BAR_WIDTH)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=14, fontweight="bold")
        ax.set_title(f"Exact M0 vs M3 by TMIN subset: {metric}", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=18)
        format_axes_code2(ax, grid_axis="y")
        move_legend_outside(ax)
        save_bar_figure(os.path.join(output_dir, f"TMIN_subset_{metric}_exact_M0_vs_M3.png"))

    pred_rows = []
    if m0_ref is not None and TMIN_COL in m0_ref["test_original_df"].columns:
        m0_test_df = m0_ref["test_original_df"].copy()
        m0_high = m0_test_df[TMIN_COL].values > TMIN_THRESHOLD
        m0_low = ~m0_high
        pred_rows.append(pd.DataFrame({"True": m0_ref["y_test_best"][m0_low], "Predicted": m0_ref["y_pred_test_best"][m0_low], "Model": "M0", "Subset": "Low TMIN test"}))
        pred_rows.append(pd.DataFrame({"True": m0_ref["y_test_best"][m0_high], "Predicted": m0_ref["y_pred_test_best"][m0_high], "Model": "M0", "Subset": "High TMIN test"}))
    pred_rows.append(pd.DataFrame({"True": y_m3[low_m3], "Predicted": pred_m3[low_m3], "Model": "M3", "Subset": "Low TMIN test"}))
    pred_rows.append(pd.DataFrame({"True": y_m3[high_m3], "Predicted": pred_m3[high_m3], "Model": "M3", "Subset": "High TMIN test"}))
    long_pred = pd.concat(pred_rows, axis=0, ignore_index=True)
    long_pred.to_excel(os.path.join(output_dir, "TMIN_subset_prediction_data_exact_M0_vs_M3.xlsx"), index=False)

    for subset_name in ["Low TMIN test", "High TMIN test"]:
        sub = long_pred[long_pred["Subset"] == subset_name].copy()
        if len(sub) >= 3:
            plt.figure(figsize=FIGSIZE["true_vs_pred"])
            ax = sns.scatterplot(data=sub, x="True", y="Predicted", hue="Model", s=110, alpha=0.75, edgecolor="k")
            min_v = float(min(sub["True"].min(), sub["Predicted"].min()))
            max_v = float(max(sub["True"].max(), sub["Predicted"].max()))
            ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", linewidth=1.5)
            ax.set_xlabel("True Values", fontsize=18, fontweight="bold")
            ax.set_ylabel("Predicted Values", fontsize=18, fontweight="bold")
            ax.set_title(subset_name, fontsize=20, fontweight="bold")
            format_axes_code2(ax, grid_axis="both")
            move_legend_outside(ax)
            safe_subset = subset_name.replace(" ", "_")
            save_bar_figure(os.path.join(output_dir, f"{safe_subset}_true_vs_predicted_exact_M0_vs_M3.png"))
    return out_df

def residual_proportion_summary(y_true, pred, model_name, subset_name, threshold_values=(0.5, 1.0, 2.0)):
    resid = np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float)
    abs_resid = np.abs(resid)
    rows = []
    for thr in threshold_values:
        rows.append({"Subset": subset_name, "Model": model_name, "Residual class": f"|Residual| <= {thr}", "Proportion": float(np.mean(abs_resid <= thr)) if len(abs_resid) else np.nan, "n": int(len(abs_resid))})
        rows.append({"Subset": subset_name, "Model": model_name, "Residual class": f"|Residual| > {thr}", "Proportion": float(np.mean(abs_resid > thr)) if len(abs_resid) else np.nan, "n": int(len(abs_resid))})
    rows.append({"Subset": subset_name, "Model": model_name, "Residual class": "Positive residual", "Proportion": float(np.mean(resid > 0)) if len(resid) else np.nan, "n": int(len(resid))})
    rows.append({"Subset": subset_name, "Model": model_name, "Residual class": "Negative residual", "Proportion": float(np.mean(resid < 0)) if len(resid) else np.nan, "n": int(len(resid))})
    return rows


def plot_residual_distribution_proportions(best_bundle, output_dir):
    """Compare residual-distribution proportions for all, low-TMIN, and high-TMIN subsets using each model's own fixed test set."""
    m0_ref = best_bundle.get("external_m0_reference", None)
    test_df = best_bundle["test_df"].copy()
    y_m3 = np.asarray(best_bundle["y_test"], dtype=float)
    pred_m3 = np.asarray(best_bundle["m3_test_pred"], dtype=float)
    high_m3 = test_df[TMIN_COL].values > TMIN_THRESHOLD
    low_m3 = ~high_m3

    rows = []
    for subset_name, mask in [
        ("All test", np.ones(len(test_df), dtype=bool)),
        ("Low TMIN test", low_m3),
        ("High TMIN test", high_m3),
    ]:
        rows.extend(residual_proportion_summary(y_m3[mask], pred_m3[mask], "M3", subset_name))

    if m0_ref is not None:
        m0_test_df = m0_ref["test_original_df"].copy()
        y_m0 = np.asarray(m0_ref["y_test_best"], dtype=float)
        pred_m0 = np.asarray(m0_ref["y_pred_test_best"], dtype=float)
        if TMIN_COL in m0_test_df.columns:
            high_m0 = m0_test_df[TMIN_COL].values > TMIN_THRESHOLD
            low_m0 = ~high_m0
        else:
            high_m0 = np.zeros(len(m0_test_df), dtype=bool)
            low_m0 = np.ones(len(m0_test_df), dtype=bool)
        for subset_name, mask in [
            ("All test", np.ones(len(m0_test_df), dtype=bool)),
            ("Low TMIN test", low_m0),
            ("High TMIN test", high_m0),
        ]:
            rows.extend(residual_proportion_summary(y_m0[mask], pred_m0[mask], "M0", subset_name))

    prop_df = pd.DataFrame(rows)
    prop_df.to_excel(os.path.join(output_dir, "residual_distribution_proportion_exact_M0_vs_M3.xlsx"), index=False)

    for subset_name in prop_df["Subset"].unique():
        sub = prop_df[prop_df["Subset"] == subset_name].copy()
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=sub, x="Residual class", y="Proportion", hue="Model", edgecolor="black")
        shrink_bar_width(ax, width=BAR_WIDTH)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion", fontsize=14, fontweight="bold")
        ax.set_title(f"Residual Proportion: {subset_name}", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        format_axes_code2(ax, grid_axis="y")
        move_legend_outside(ax)
        save_bar_figure(os.path.join(output_dir, f"residual_proportion_{subset_name.replace(' ', '_')}_exact_M0_vs_M3.png"))
    return prop_df

def plot_constrained_pdp(best_bundle, output_dir):
    residual_model = best_bundle["residual_model"]
    feature_cols = best_bundle["residual_feature_cols"]
    test_df = best_bundle["test_df"].copy()
    high_df = test_df[test_df[TMIN_COL] > TMIN_THRESHOLD].copy()

    if high_df.empty or PROTEIN_RISK_COL not in feature_cols:
        warning_df = pd.DataFrame([{"warning": "No high-TMIN samples or Protein_Heat_Risk not in residual feature columns."}])
        warning_df.to_excel(os.path.join(output_dir, "constrained_PDP_warning.xlsx"), index=False)
        return warning_df

    feature_df = build_r5_residual_feature_df(high_df, best_bundle["base_weather_features"], PREDICTED_PROTEIN_COL)[feature_cols]
    x = feature_df[PROTEIN_RISK_COL].values
    x_min = float(np.nanpercentile(x, 5))
    x_max = float(np.nanpercentile(x, 95))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
    grid = np.linspace(x_min, x_max, 80)

    pdp_values = []
    for gx in grid:
        tmp = feature_df.copy()
        tmp[PROTEIN_RISK_COL] = gx
        pdp_values.append(float(np.mean(residual_model.predict(tmp.values))))

    pdp_df = pd.DataFrame({PROTEIN_RISK_COL: grid, "Mean constrained residual correction": pdp_values})
    pdp_df["First difference"] = pdp_df["Mean constrained residual correction"].diff()
    pdp_df.to_excel(os.path.join(output_dir, "constrained_PDP_Protein_Heat_Risk_high_TMIN.xlsx"), index=False)

    plt.figure(figsize=FIGSIZE["pdp"])
    ax = sns.lineplot(data=pdp_df, x=PROTEIN_RISK_COL, y="Mean constrained residual correction", linewidth=3, color="blue")
    ax.scatter(pdp_df[PROTEIN_RISK_COL], pdp_df["Mean constrained residual correction"], s=18, color="blue", alpha=0.45)
    ax.set_xlabel(PROTEIN_RISK_COL, fontsize=18, fontweight="bold")
    ax.set_ylabel("Mean residual correction", fontsize=18, fontweight="bold")
    ax.set_title("Constrained PDP", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(output_dir, "constrained_PDP_Protein_Heat_Risk_high_TMIN.png"))

    plt.figure(figsize=FIGSIZE["pdp"])
    ax = sns.histplot(pdp_df["First difference"].dropna(), bins=20, kde=True, alpha=0.65, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("First difference of PDP", fontsize=18, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=18, fontweight="bold")
    ax.set_title("Constraint Check", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, "constrained_PDP_first_difference_check.png"))
    return pdp_df


def plot_alpha_rationality_validation(best_bundle, output_dir):
    df = best_bundle["test_df"].copy()
    df["True"] = best_bundle["y_test"]
    df["M3 base prediction"] = best_bundle["internal_m0_test_pred"]
    # External M0 is computed independently on its own DATA_PATH and is not row-aligned to this M3 test set.
    # Therefore, alpha validation is reported as a mechanism diagnostic for the M3 base prediction, not as an M0 result.
    df["Raw correction"] = best_bundle["raw_test_correction"]
    df["Alpha"] = best_bundle["test_alpha"]
    df["Applied correction"] = df["Alpha"] * df["Raw correction"]
    df["Observed M3-base residual"] = df["True"] - df["M3 base prediction"]
    df["M3 residual"] = df["True"] - best_bundle["m3_test_pred"]
    df["Correction direction correct"] = np.sign(df["Applied correction"]) == np.sign(df["Observed M3-base residual"])
    df["Absolute residual reduction"] = np.abs(df["Observed M3-base residual"]) - np.abs(df["M3 residual"])
    df.to_excel(os.path.join(output_dir, "alpha_rationality_validation_data.xlsx"), index=False)

    plot_specs = [
        ("M3 base prediction", "Alpha", "alpha_vs_M3_base_prediction.png", "Alpha and M3 Base Prediction"),
        ("Alpha", "Absolute residual reduction", "error_reduction_vs_alpha.png", "Error Reduction and Alpha"),
        ("Alpha", "Observed M3-base residual", "observed_M3_base_residual_vs_alpha.png", "Residual and Alpha"),
        ("Alpha", "Applied correction", "applied_correction_vs_alpha.png", "Applied Correction and Alpha"),
    ]
    for x_col, y_col, fname, title in plot_specs:
        plt.figure(figsize=FIGSIZE["correction_diagnostic"])
        ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue=TMIN_COL if TMIN_COL in df.columns else None, s=100, alpha=0.75, edgecolor="k")
        if y_col != "Alpha":
            ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel(x_col, fontsize=18, fontweight="bold")
        ax.set_ylabel(y_col, fontsize=18, fontweight="bold")
        ax.set_title(title, fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        if TMIN_COL in df.columns:
            move_legend_outside(ax, title=TMIN_COL)
            save_bar_figure(os.path.join(output_dir, fname))
        else:
            save_current_figure(os.path.join(output_dir, fname))

    direction_summary = df.groupby("Correction direction correct", dropna=False).size().reset_index(name="n")
    direction_summary["Proportion"] = direction_summary["n"] / direction_summary["n"].sum()
    direction_summary.to_excel(os.path.join(output_dir, "alpha_direction_correctness_summary.xlsx"), index=False)

    plt.figure(figsize=FIGSIZE["performance_bar"])
    ax = sns.barplot(data=direction_summary, x="Correction direction correct", y="Proportion", edgecolor="black")
    shrink_bar_width(ax, width=BAR_WIDTH)
    ax.set_xlabel("Correction direction correct", fontsize=14, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=14, fontweight="bold")
    ax.set_title("Alpha-scaled Correction Direction", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, "alpha_direction_correctness_proportion.png"))
    return df


def plot_alpha_search_diagnostics(best_bundle, output_dir):
    alpha_df = best_bundle.get("alpha_df", None)
    if alpha_df is None or not isinstance(alpha_df, pd.DataFrame) or alpha_df.empty:
        return None

    alpha_df = alpha_df.copy()
    alpha_df.to_excel(os.path.join(output_dir, "selected_model_alpha_search_all_candidates.xlsx"), index=False)

    pivot_base_amp = alpha_df.pivot_table(index="alpha_base", columns="alpha_amp", values="selection_score", aggfunc="max")
    pivot_base_amp.to_excel(os.path.join(output_dir, "alpha_search_base_amp_heatmap_data.xlsx"))
    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(pivot_base_amp, cmap="Blues", linewidths=0.4, linecolor="white")
    ax.set_xlabel("Alpha amp", fontsize=14, fontweight="bold")
    ax.set_ylabel("Alpha base", fontsize=14, fontweight="bold")
    ax.set_title("Alpha Search", fontsize=16, fontweight="bold")
    save_current_figure(os.path.join(output_dir, "alpha_search_base_amp_heatmap.png"))

    pivot_slope_center = alpha_df.pivot_table(index="alpha_slope", columns="alpha_center_quantile", values="selection_score", aggfunc="max")
    pivot_slope_center.to_excel(os.path.join(output_dir, "alpha_search_slope_center_heatmap_data.xlsx"))
    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(pivot_slope_center, cmap="Blues", linewidths=0.4, linecolor="white")
    ax.set_xlabel("Center quantile", fontsize=14, fontweight="bold")
    ax.set_ylabel("Alpha slope", fontsize=14, fontweight="bold")
    ax.set_title("Alpha Search", fontsize=16, fontweight="bold")
    save_current_figure(os.path.join(output_dir, "alpha_search_slope_center_heatmap.png"))

    top_alpha = alpha_df.sort_values(["selection_score", "student_test_RMSE"], ascending=[False, True]).head(15).copy()
    top_alpha["Candidate"] = [f"A{i+1}" for i in range(len(top_alpha))]
    top_alpha.to_excel(os.path.join(output_dir, "top15_alpha_candidates.xlsx"), index=False)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=top_alpha, x="selection_score", y="Candidate", edgecolor="black")
    shrink_bar_height(ax)
    ax.set_xlabel("Selection score", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title("Top Alpha Candidates", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="x")
    save_current_figure(os.path.join(output_dir, "top15_alpha_candidates.png"))
    return alpha_df


def plot_importance_outputs(best_bundle, output_dir):
    outputs = {}
    model = best_bundle["internal_m0_model"]
    scaler = best_bundle["internal_m0_scaler"]
    weather_cols = best_bundle["base_weather_features"]
    X_test_weather = best_bundle["test_df"][weather_cols].values
    y_test = best_bundle["y_test"]

    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({"Feature": weather_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        imp_df.to_excel(os.path.join(output_dir, "M3_base_RF_feature_importance.xlsx"), index=False)
        outputs["M3_base_RF_importance"] = imp_df

        top_df = imp_df.head(TOP_K_FEATURES)
        values = top_df["Importance"].values
        names = top_df["Feature"].values
        order = np.argsort(values)
        plt.figure(figsize=FIGSIZE["importance"])
        ax = plt.gca()
        ax.barh(np.arange(len(values)), values[order], color=plt.cm.Blues(np.linspace(0.45, 0.95, len(values))), edgecolor="black", linewidth=1.2, height=BARH_HEIGHT)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(names[order], fontsize=14, fontweight="bold")
        ax.set_xlabel("RF feature importance", fontsize=14, fontweight="bold")
        ax.set_title("M3 Base RF Feature Importance", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="x")
        save_current_figure(os.path.join(output_dir, "M3_base_RF_feature_importance.png"))

    try:
        perm = permutation_importance(model, scaler.transform(X_test_weather), y_test, scoring="r2", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
        perm_df = pd.DataFrame({
            "Feature": weather_cols,
            "Permutation importance mean": perm.importances_mean,
            "Permutation importance std": perm.importances_std,
        }).sort_values("Permutation importance mean", ascending=False)
        perm_df.to_excel(os.path.join(output_dir, "M3_base_permutation_importance.xlsx"), index=False)
        outputs["M3_base_permutation"] = perm_df
    except Exception as e:
        pd.DataFrame([{"stage": "Permutation importance", "error": str(e)}]).to_excel(os.path.join(output_dir, "M3_base_permutation_importance_error.xlsx"), index=False)

    if shap is not None:
        try:
            X_scaled = scaler.transform(X_test_weather)
            explainer = shap.TreeExplainer(model)
            shap_values = np.asarray(explainer.shap_values(X_scaled))
            shap_df = pd.DataFrame({"Feature": weather_cols, "SHAP Mean Importance": np.abs(shap_values).mean(axis=0)}).sort_values("SHAP Mean Importance", ascending=False)
            shap_df.to_excel(os.path.join(output_dir, "M3_base_SHAP_importance.xlsx"), index=False)
            outputs["M3_base_SHAP"] = shap_df
        except Exception as e:
            pd.DataFrame([{"stage": "SHAP", "error": str(e)}]).to_excel(os.path.join(output_dir, "M3_base_SHAP_error.xlsx"), index=False)
    return outputs


def plot_model_screening_diagnostics(result_df, output_dir):
    if result_df is None or result_df.empty:
        return None

    family_summary = (
        result_df.groupby("Model family", dropna=False)
        .agg(
            n_models=("Version", "count"),
            max_test_R2=("M3 test R2", "max"),
            mean_test_R2=("M3 test R2", "mean"),
            min_test_RMSE=("M3 test RMSE", "min"),
            mean_test_RMSE=("M3 test RMSE", "mean"),
            max_delta_R2_vs_M0=("Delta test R2 M3_minus_external_M0", "max"),
            mean_delta_R2_vs_M0=("Delta test R2 M3_minus_external_M0", "mean"),
        )
        .reset_index()
    )
    family_summary.to_excel(os.path.join(output_dir, "model_family_screening_summary.xlsx"), index=False)

    plt.figure(figsize=FIGSIZE["performance_bar"])
    ax = sns.boxplot(data=result_df, x="Model family", y="M3 test R2", showfliers=False)
    sns.stripplot(data=result_df, x="Model family", y="M3 test R2", color="black", alpha=0.45, size=5)
    ax.set_xlabel("")
    ax.set_ylabel("Test R²", fontsize=14, fontweight="bold")
    ax.set_title("Model Screening", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, "model_family_test_R2_distribution.png"))

    plt.figure(figsize=FIGSIZE["performance_bar"])
    ax = sns.scatterplot(data=result_df, x="M3 test RMSE", y="M3 test R2", hue="Model family", s=90, edgecolor="black", alpha=0.80)
    ax.set_xlabel("Test RMSE", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test R²", fontsize=14, fontweight="bold")
    ax.set_title("Accuracy Trade-off", fontsize=16, fontweight="bold")
    format_axes_code2(ax, grid_axis="both")
    save_current_figure(os.path.join(output_dir, "model_family_R2_RMSE_tradeoff.png"))
    return family_summary


def plot_weight_outputs(best_record, teacher_clean, test_df, output_dir):
    scheme = best_record.get("Weight scheme", "none")
    beta = best_record.get("Weight beta", 0.0)
    location = best_record.get("Weight location", "none")
    if scheme == "none" or pd.isna(beta):
        return None

    residual_weight = build_sample_weight(teacher_clean, scheme=scheme, beta=float(beta), protein_col=PROTEIN_COL)
    weight_df = pd.DataFrame({
        "Dataset": "Teacher",
        "Weight": residual_weight,
        TMIN_COL: teacher_clean[TMIN_COL].values,
        TMIN_EXCESS_COL: teacher_clean[TMIN_EXCESS_COL].values,
        PROTEIN_COL: teacher_clean[PROTEIN_COL].values,
        PROTEIN_RISK_COL: teacher_clean[PROTEIN_RISK_COL].values if PROTEIN_RISK_COL in teacher_clean.columns else np.nan,
    })

    if location == "residual_and_alpha":
        alpha_weight = build_sample_weight(test_df, scheme=scheme, beta=float(beta), protein_col=PREDICTED_PROTEIN_COL)
        alpha_df = pd.DataFrame({
            "Dataset": "Test",
            "Weight": alpha_weight,
            TMIN_COL: test_df[TMIN_COL].values,
            TMIN_EXCESS_COL: test_df[TMIN_EXCESS_COL].values,
            PREDICTED_PROTEIN_COL: test_df[PREDICTED_PROTEIN_COL].values,
            PROTEIN_RISK_COL: test_df[PROTEIN_RISK_COL].values,
        })
        weight_df = pd.concat([weight_df, alpha_df], axis=0, ignore_index=True)

    weight_df.to_excel(os.path.join(output_dir, "best_M3_weight_distribution_data.xlsx"), index=False)

    plt.figure(figsize=FIGSIZE["weight"])
    ax = sns.histplot(data=weight_df, x="Weight", hue="Dataset", bins=15, alpha=0.65, edgecolor="black")
    plt.xlabel("Sample weight", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("Sample Weight Distribution", fontsize=20, fontweight="bold")
    format_axes_code2(ax, grid_axis="y")
    save_current_figure(os.path.join(output_dir, "best_M3_weight_distribution.png"))
    return weight_df


def plot_weight_beta_effects(result_df, output_dir):
    """
    Plot beta-response curves for the weighted R5CW search.

    These figures show whether increasing beta improves or worsens M3 test
    performance under different weight schemes and weight locations.
    """
    if result_df is None or result_df.empty or "Model family" not in result_df.columns:
        return pd.DataFrame()

    r5cw_df = result_df[result_df["Model family"] == "R5CW"].copy()
    if r5cw_df.empty:
        warning_df = pd.DataFrame([{"warning": "No R5CW records available for beta-effect plotting."}])
        warning_df.to_excel(os.path.join(output_dir, "weight_beta_effect_warning.xlsx"), index=False)
        return warning_df

    metric_cols = [
        "M3 test R2", "M3 test RMSE", "M3 test MSE", "M3 test MAE",
        "Delta test R2 M3_minus_R5C", "Delta test RMSE M3_minus_R5C", "Delta test MAE M3_minus_R5C",
        "Delta test R2 M3_minus_external_M0", "Delta test RMSE M3_minus_external_M0", "Delta test MAE M3_minus_external_M0",
    ]
    metric_cols = [c for c in metric_cols if c in r5cw_df.columns]
    beta_effect_df = r5cw_df[["Version", "Weight scheme", "Weight beta", "Weight location"] + metric_cols].copy()
    beta_effect_df.to_excel(os.path.join(output_dir, "weight_beta_effect_metrics.xlsx"), index=False)

    for metric in ["M3 test R2", "M3 test RMSE", "M3 test MSE", "M3 test MAE", "Delta test R2 M3_minus_R5C"]:
        if metric not in r5cw_df.columns:
            continue
        plot_df = r5cw_df.dropna(subset=[metric, "Weight beta"]).copy()
        if plot_df.empty:
            continue
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=plot_df,
            x="Weight beta",
            y=metric,
            hue="Weight scheme",
            style="Weight location",
            markers=True,
            dashes=False,
            linewidth=2.2,
            markersize=7,
            errorbar=None,
        )
        ax.set_xlabel("Weight beta", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric.replace("M3 test ", "Test "), fontsize=14, fontweight="bold")
        ax.set_title(f"Beta Effect: {metric}", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        move_legend_outside(ax)
        save_bar_figure(os.path.join(output_dir, f"beta_effect_{metric.replace(' ', '_').replace('/', '_')}.png"))

    return beta_effect_df


def plot_weight_scheme_beta_heatmaps(result_df, output_dir):
    """
    Plot weight scheme by beta heatmaps for each weight location.

    Rows represent weight schemes, columns represent beta values, and cell values
    represent model performance or improvement over R5C.
    """
    if result_df is None or result_df.empty or "Model family" not in result_df.columns:
        return {}

    r5cw_df = result_df[result_df["Model family"] == "R5CW"].copy()
    outputs = {}
    if r5cw_df.empty:
        warning_df = pd.DataFrame([{"warning": "No R5CW records available for scheme-beta heatmaps."}])
        warning_df.to_excel(os.path.join(output_dir, "weight_scheme_beta_heatmap_warning.xlsx"), index=False)
        outputs["warning"] = warning_df
        return outputs

    heatmap_metrics = [
        "M3 test R2",
        "M3 test RMSE",
        "M3 test MAE",
        "Delta test R2 M3_minus_R5C",
        "Delta test RMSE M3_minus_R5C",
        "Delta test MAE M3_minus_R5C",
        "Delta test R2 M3_minus_external_M0",
    ]
    heatmap_metrics = [m for m in heatmap_metrics if m in r5cw_df.columns]

    for location in sorted(r5cw_df["Weight location"].dropna().unique()):
        loc_df = r5cw_df[r5cw_df["Weight location"] == location].copy()
        if loc_df.empty:
            continue
        for metric in heatmap_metrics:
            pivot = loc_df.pivot_table(
                index="Weight scheme",
                columns="Weight beta",
                values=metric,
                aggfunc="mean",
            )
            if pivot.empty:
                continue
            safe_metric = metric.replace(" ", "_").replace("/", "_")
            pivot_path = os.path.join(output_dir, f"weight_scheme_beta_heatmap_{location}_{safe_metric}.xlsx")
            pivot.to_excel(pivot_path)
            outputs[f"{location}_{safe_metric}"] = pivot.reset_index()

            plt.figure(figsize=FIGSIZE["heatmap"])
            ax = sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".3f", linewidths=0.5, linecolor="white")
            ax.set_xlabel("Weight beta", fontsize=14, fontweight="bold")
            ax.set_ylabel("Weight scheme", fontsize=14, fontweight="bold")
            ax.set_title(f"Weight scheme × beta: {metric}\n{location}", fontsize=16, fontweight="bold")
            save_current_figure(os.path.join(output_dir, f"weight_scheme_beta_heatmap_{location}_{safe_metric}.png"))

    return outputs


def plot_weight_tmin_effects(result_df, output_dir):
    """
    Plot weighted-model effects under low- and high-TMIN test subsets.

    This function uses the subset metrics added to each R5C and R5CW record.
    R5C is the unweighted conditional-constraint reference, while R5CW records
    show the effect of different weight schemes, beta values, and weight locations.
    """
    if result_df is None or result_df.empty:
        return pd.DataFrame()

    required_cols = ["M3 Low TMIN test R2", "M3 High TMIN test R2"]
    if not all(c in result_df.columns for c in required_cols):
        warning_df = pd.DataFrame([{"warning": "TMIN subset columns are not available in result_df."}])
        warning_df.to_excel(os.path.join(output_dir, "weight_TMIN_effect_warning.xlsx"), index=False)
        return warning_df

    keep_families = ["R5C", "R5CW"]
    df = result_df[result_df["Model family"].isin(keep_families)].copy()
    if df.empty:
        warning_df = pd.DataFrame([{"warning": "No R5C or R5CW records available for TMIN subset weight-effect plotting."}])
        warning_df.to_excel(os.path.join(output_dir, "weight_TMIN_effect_warning.xlsx"), index=False)
        return warning_df

    long_rows = []
    for _, row in df.iterrows():
        for subset in ["Low TMIN", "High TMIN"]:
            for metric in ["R2", "RMSE", "MSE", "MAE"]:
                col = f"M3 {subset} test {metric}"
                if col in df.columns:
                    long_rows.append({
                        "Model family": row.get("Model family", np.nan),
                        "Version": row.get("Version", np.nan),
                        "Weight scheme": row.get("Weight scheme", "none"),
                        "Weight beta": row.get("Weight beta", 0.0),
                        "Weight location": row.get("Weight location", "none"),
                        "Subset": subset,
                        "Metric": metric,
                        "Value": row.get(col, np.nan),
                        "n": row.get(f"M3 {subset} test n", np.nan),
                    })
    subset_effect_df = pd.DataFrame(long_rows)
    subset_effect_df.to_excel(os.path.join(output_dir, "weight_TMIN_subset_effect_metrics.xlsx"), index=False)

    # Add delta versus the unweighted R5C reference within each subset and metric.
    baseline = subset_effect_df[subset_effect_df["Model family"] == "R5C"][["Subset", "Metric", "Value"]].dropna().copy()
    baseline = baseline.rename(columns={"Value": "R5C baseline"})
    baseline = baseline.groupby(["Subset", "Metric"], as_index=False)["R5C baseline"].mean()
    subset_effect_df = subset_effect_df.merge(baseline, on=["Subset", "Metric"], how="left")
    subset_effect_df["Delta vs R5C"] = subset_effect_df["Value"] - subset_effect_df["R5C baseline"]
    subset_effect_df.to_excel(os.path.join(output_dir, "weight_TMIN_subset_effect_metrics_with_delta_vs_R5C.xlsx"), index=False)

    weighted_df = subset_effect_df[subset_effect_df["Model family"] == "R5CW"].copy()
    for location in sorted(weighted_df["Weight location"].dropna().unique()):
        loc_df = weighted_df[weighted_df["Weight location"] == location].copy()
        if loc_df.empty:
            continue
        for metric in ["R2", "RMSE", "MAE"]:
            plot_df = loc_df[loc_df["Metric"] == metric].dropna(subset=["Value", "Weight beta"]).copy()
            if plot_df.empty:
                continue
            plt.figure(figsize=(8, 5))
            ax = sns.lineplot(
                data=plot_df,
                x="Weight beta",
                y="Value",
                hue="Weight scheme",
                style="Subset",
                markers=True,
                dashes=False,
                linewidth=2.2,
                markersize=7,
                errorbar=None,
            )
            base_sub = baseline[baseline["Metric"] == metric]
            for _, base_row in base_sub.iterrows():
                if base_row["Subset"] == "Low TMIN":
                    ax.axhline(base_row["R5C baseline"], color="gray", linestyle="--", linewidth=1.2, alpha=0.75)
                elif base_row["Subset"] == "High TMIN":
                    ax.axhline(base_row["R5C baseline"], color="black", linestyle=":", linewidth=1.2, alpha=0.75)
            ax.set_xlabel("Weight beta", fontsize=14, fontweight="bold")
            ax.set_ylabel(metric, fontsize=14, fontweight="bold")
            ax.set_title(f"Weighted effect by TMIN subset: {metric}\n{location}", fontsize=16, fontweight="bold")
            format_axes_code2(ax, grid_axis="both")
            move_legend_outside(ax)
            save_bar_figure(os.path.join(output_dir, f"weight_TMIN_subset_{location}_{metric}.png"))

            delta_df = loc_df[loc_df["Metric"] == metric].dropna(subset=["Delta vs R5C", "Weight beta"]).copy()
            if not delta_df.empty:
                plt.figure(figsize=(8, 5))
                ax = sns.lineplot(
                    data=delta_df,
                    x="Weight beta",
                    y="Delta vs R5C",
                    hue="Weight scheme",
                    style="Subset",
                    markers=True,
                    dashes=False,
                    linewidth=2.2,
                    markersize=7,
                    errorbar=None,
                )
                ax.axhline(0, color="r", linestyle="--", linewidth=1.5)
                ax.set_xlabel("Weight beta", fontsize=14, fontweight="bold")
                ax.set_ylabel(f"Delta {metric} vs R5C", fontsize=14, fontweight="bold")
                ax.set_title(f"Weighted improvement by TMIN subset: {metric}\n{location}", fontsize=16, fontweight="bold")
                format_axes_code2(ax, grid_axis="both")
                move_legend_outside(ax)
                save_bar_figure(os.path.join(output_dir, f"weight_TMIN_subset_delta_vs_R5C_{location}_{metric}.png"))

        # Heatmaps for low and high TMIN R2 by scheme and beta.
        for subset in ["Low TMIN", "High TMIN"]:
            for metric in ["R2", "RMSE", "MAE"]:
                plot_df = loc_df[(loc_df["Subset"] == subset) & (loc_df["Metric"] == metric)].copy()
                pivot = plot_df.pivot_table(index="Weight scheme", columns="Weight beta", values="Value", aggfunc="mean")
                if pivot.empty:
                    continue
                safe_subset = subset.replace(" ", "_")
                pivot.to_excel(os.path.join(output_dir, f"weight_{safe_subset}_{location}_{metric}_scheme_beta_heatmap_data.xlsx"))
                plt.figure(figsize=FIGSIZE["heatmap"])
                ax = sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".3f", linewidths=0.5, linecolor="white")
                ax.set_xlabel("Weight beta", fontsize=14, fontweight="bold")
                ax.set_ylabel("Weight scheme", fontsize=14, fontweight="bold")
                ax.set_title(f"{subset} weighted effect: {metric}\n{location}", fontsize=16, fontweight="bold")
                save_current_figure(os.path.join(output_dir, f"weight_{safe_subset}_{location}_{metric}_scheme_beta_heatmap.png"))

    return subset_effect_df



# =========================================================
# 8. Training-fraction sensitivity
# =========================================================
def train_quick_m3_for_fraction(train_df, test_df, y_train, y_test, base_weather_features, teacher_clean, best_weighted_record, cv, random_state):
    internal_m0_model, internal_m0_scaler, _, internal_m0_train_pred, internal_m0_test_pred = train_rf_on_fixed_split(
        train_df[base_weather_features].values,
        y_train,
        test_df[base_weather_features].values,
        cv=cv,
        random_state=random_state,
        n_iter=M0_N_ITER,
    )

    scheme = best_weighted_record.get("Weight scheme", "none")
    beta = float(best_weighted_record.get("Weight beta", 0.0))
    location = best_weighted_record.get("Weight location", "none")

    residual_weight = None
    alpha_weight = None
    if scheme != "none":
        residual_weight = build_sample_weight(teacher_clean, scheme=scheme, beta=beta, protein_col=PROTEIN_COL)
        if location == "residual_and_alpha":
            alpha_weight = build_sample_weight(test_df, scheme=scheme, beta=beta, protein_col=PREDICTED_PROTEIN_COL)

    residual_model, _, feature_cols, _ = fit_r5_conditional_residual_model(
        teacher_clean,
        base_weather_features,
        internal_m0_model,
        internal_m0_scaler,
        sample_weight=residual_weight,
        task_suffix="fraction_sensitivity",
        random_state=random_state,
    )
    raw_train, raw_test, _, _ = make_r5_residual_predictions(residual_model, feature_cols, train_df, test_df, base_weather_features)
    alpha_metric = "weighted_r2" if alpha_weight is not None else "r2"

    rec, _ = evaluate_m3(
        version="M3_fraction_sensitivity",
        y_train=y_train,
        y_test=y_test,
        internal_m0_train_pred=internal_m0_train_pred,
        internal_m0_test_pred=internal_m0_test_pred,
        raw_train_correction=raw_train,
        raw_test_correction=raw_test,
        alpha_selection_weight=alpha_weight,
        alpha_selection_metric=alpha_metric,
    )
    return rec



def run_training_fraction_sensitivity(aligned_m3_df, base_weather_features,
                                      original_target_col, teacher_clean, best_weighted_record,
                                      output_dir, m0_quantity_df=None):
    """
    Training-quantity comparison after independent model computation.

    M0 is not fitted on M3 data. M0 results are supplied by
    compute_exact_m0_training_quantity_results(), which reads M0_DATA_PATH and applies Z3CV7.
    M3 is computed from the selected M3 cleaned sample set with its own fixed 30% test set.
    The final table only stacks the independently computed M0 and M3 results for comparison.
    """
    records = []

    if m0_quantity_df is not None and isinstance(m0_quantity_df, pd.DataFrame) and not m0_quantity_df.empty:
        records.extend(m0_quantity_df.to_dict("records"))

    n_m3 = len(aligned_m3_df)
    m3_indices = np.arange(n_m3)
    m3_train_pool_idx, m3_test_idx = train_test_split(
        m3_indices,
        test_size=FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS,
        random_state=RANDOM_STATE,
    )
    fixed_test_df_m3 = aligned_m3_df.iloc[m3_test_idx].copy()
    y_test_m3 = fixed_test_df_m3[original_target_col].values

    for frac in TRAIN_FRACTION_LIST:
        n_train_m3 = int(round(n_m3 * frac))
        n_train_m3 = min(n_train_m3, len(m3_train_pool_idx))
        if n_train_m3 < max(8, min(STUDENT_CV_VALUES)):
            print(f"M3 training fraction {frac}: too few training samples; skipped.")
            continue

        for rep in range(TRAIN_FRACTION_REPEATS):
            if np.isclose(frac, 1.0 - FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS) and rep == 0:
                seed = RANDOM_STATE
                train_idx_m3 = m3_train_pool_idx
            else:
                seed = RANDOM_STATE + 1000 + int(frac * 100) * 10 + rep
                rng = np.random.default_rng(seed)
                train_idx_m3 = rng.choice(m3_train_pool_idx, size=n_train_m3, replace=False)

            train_df_m3 = aligned_m3_df.iloc[train_idx_m3].copy()
            y_train_m3 = train_df_m3[original_target_col].values
            cv3 = min(STUDENT_CV_VALUES[0], max(2, len(train_df_m3) // 3))
            if cv3 >= len(train_df_m3):
                cv3 = max(2, len(train_df_m3) - 1)

            m3_rec = train_quick_m3_for_fraction(
                train_df=train_df_m3,
                test_df=fixed_test_df_m3,
                y_train=y_train_m3,
                y_test=y_test_m3,
                base_weather_features=base_weather_features,
                teacher_clean=teacher_clean,
                best_weighted_record=best_weighted_record,
                cv=cv3,
                random_state=seed,
            )
            if np.isclose(frac, 1.0 - FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS) and rep == 0:
                m3_r2 = best_weighted_record.get("M3 test R2", m3_rec["M3 test R2"])
                m3_rmse = best_weighted_record.get("M3 test RMSE", m3_rec["M3 test RMSE"])
                m3_mae = best_weighted_record.get("M3 test MAE", m3_rec["M3 test MAE"])
                m3_mse = best_weighted_record.get("M3 test MSE", np.nan)
            else:
                m3_r2 = m3_rec["M3 test R2"]
                m3_rmse = m3_rec["M3 test RMSE"]
                m3_mae = m3_rec["M3 test MAE"]
                m3_mse = m3_rec.get("M3 test MSE", np.nan)

            records.append({
                "Model": "M3",
                "Training fraction": float(frac),
                "Repeat": int(rep + 1),
                "n_train": int(n_train_m3),
                "n_test": int(len(m3_test_idx)),
                "unused_fraction": float(max(0.0, 1.0 - FIXED_TEST_SIZE_FOR_FRACTION_ANALYSIS - frac)),
                "R2": float(m3_r2),
                "RMSE": float(m3_rmse),
                "MSE": float(m3_mse) if np.isfinite(m3_mse) else np.nan,
                "MAE": float(m3_mae),
                "M3_z_score_threshold": STUDENT_Z_THRESHOLDS[0],
                "M3_cv": cv3,
            })

    frac_df = pd.DataFrame(records)
    frac_df.to_excel(os.path.join(output_dir, "training_quantity_independent_M0_Z3CV7_vs_M3.xlsx"), index=False)

    for metric in ["R2", "RMSE", "MAE"]:
        plt.figure(figsize=FIGSIZE["box"])
        ax = sns.boxplot(data=frac_df, x="Training fraction", y=metric, hue="Model", width=0.36, showfliers=False)
        sns.stripplot(data=frac_df, x="Training fraction", y=metric, hue="Model", dodge=True, color="black", alpha=0.30, size=3)
        handles, labels = ax.get_legend_handles_labels()
        keep = []
        keep_labels = []
        for h, lab in zip(handles, labels):
            if lab not in keep_labels:
                keep.append(h)
                keep_labels.append(lab)
        ax.legend(keep, keep_labels, fontsize=12)
        ax.set_xlabel("Training fraction of all samples", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric, fontsize=14, fontweight="bold")
        ax.set_title(f"Training Quantity Sensitivity: {metric}", fontsize=16, fontweight="bold")
        format_axes_code2(ax, grid_axis="y")
        save_bar_figure(os.path.join(output_dir, f"training_quantity_sensitivity_{metric}_boxplot.png"))
    return frac_df


# =========================================================
# 9. Main workflow
# =========================================================
def main():
    out_dirs = build_output_dirs(OUTPUT_DIR)

    # Independent M0 workflow. M0 is computed only from its own DATA_PATH under the fixed Z3CV7 condition.
    # It is not fitted, filtered, split, or evaluated on the M3 dataset.
    m0_z3cv7_reference = train_exact_m0_z3cv7_from_user_script(
        M0_DATA_PATH,
        output_dir=out_dirs["tables"],
    )
    m0_reference = m0_z3cv7_reference
    m0_quantity_df = compute_exact_m0_training_quantity_results(
        M0_DATA_PATH,
        output_dir=out_dirs["training_fraction"],
    )

    teacher_raw = add_tmin_threshold_features(clean_column_names(pd.read_excel(TEACHER_DATA_PATH)))
    original_raw = add_tmin_threshold_features(clean_column_names(pd.read_excel(ORIGINAL_DATA_PATH)))

    print("\nTeacher columns:")
    print(teacher_raw.columns.tolist())
    print("\nM3-base columns:")
    print(original_raw.columns.tolist())
    print("\nIndependent M0 DATA_PATH:")
    print(M0_DATA_PATH)

    teacher_numeric = teacher_raw.select_dtypes(include=[np.number]).copy()
    original_numeric = original_raw.select_dtypes(include=[np.number]).copy()

    original_target_col = ORIGINAL_TARGET_COL if ORIGINAL_TARGET_COL in original_numeric.columns else original_numeric.columns[0]

    excluded_teacher = set([TARGET_COL, PROTEIN_COL] + KNOWLEDGE_ONLY_COLS)
    excluded_original = set([original_target_col] + KNOWLEDGE_ONLY_COLS)

    teacher_weather_cols = [c for c in teacher_numeric.columns if c not in excluded_teacher]
    original_weather_cols = [c for c in original_numeric.columns if c not in excluded_original]

    base_weather_features = [c for c in original_weather_cols if c in teacher_weather_cols]

    if len(base_weather_features) == 0:
        raise ValueError("No common numeric weather features were found for M3.")
    if TMIN_COL not in base_weather_features:
        raise ValueError(f"{TMIN_COL} must be in M3 base weather features.")

    pd.DataFrame({"M3_base_weather_features": base_weather_features}).to_excel(os.path.join(out_dirs["tables"], "M3_base_weather_features.xlsx"), index=False)

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
    protein_oof_metrics = calculate_metrics(teacher_clean[PROTEIN_COL].values, oof_protein, "Protein OOF prediction")

    aligned_m3_df = original_numeric[[original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    aligned_m3_df[PREDICTED_PROTEIN_COL] = protein_model.predict(protein_scaler.transform(aligned_m3_df[base_weather_features].values))
    aligned_m3_df = add_protein_heat_risk(aligned_m3_df, PREDICTED_PROTEIN_COL)

    print("\nM3 student valid n:", len(aligned_m3_df))

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

    selected_clean_m3_for_fraction = None

    for z_thr in STUDENT_Z_THRESHOLDS:
        student_clean_m3, student_outliers = remove_outliers_by_zscore(
            aligned_m3_df,
            base_weather_features,
            z_thr,
        )
        student_clean_m3 = student_clean_m3.reset_index(drop=True)

        if len(student_clean_m3) < 30:
            print(f"Z-score={z_thr}: too few samples after cleaning; skipped.")
            continue

        y_m3 = student_clean_m3[original_target_col].values

        train_idx, test_idx = train_test_split(
            np.arange(len(student_clean_m3)),
            test_size=STUDENT_TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        train_df_m3 = student_clean_m3.iloc[train_idx].copy()
        test_df_m3 = student_clean_m3.iloc[test_idx].copy()

        X_train_m3 = train_df_m3[base_weather_features].values
        X_test_m3 = test_df_m3[base_weather_features].values

        y_train_m3 = y_m3[train_idx]
        y_test_m3 = y_m3[test_idx]

        for cv in STUDENT_CV_VALUES:
            if cv >= len(X_train_m3):
                continue

            print(f"\nTraining M3 model grid: Z-score={z_thr}, CV={cv}")

            external_m0_model = m0_z3cv7_reference["model"]
            external_m0_scaler = m0_z3cv7_reference["scaler"]
            external_m0_search = m0_z3cv7_reference["search"]
            external_m0_search.reference_m0_train_r2_ = m0_z3cv7_reference["train_R2"]
            external_m0_search.reference_m0_test_r2_ = m0_z3cv7_reference["test_R2"]
            external_m0_search.reference_m0_overfit_gap_ = m0_z3cv7_reference["abs_train_test_R2_gap"]
            external_m0_search.reference_m0_pass_overfit_filter_ = m0_z3cv7_reference["pass_overfit_filter"]
            external_m0_train_metrics = {
                "External M0 train R2": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train R2"],
                "External M0 train RMSE": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train RMSE"],
                "External M0 train MSE": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train MSE"],
                "External M0 train MAE": m0_z3cv7_reference["train_metrics"]["M0 Z3CV7 train MAE"],
            }
            external_m0_test_metrics = {
                "External M0 test R2": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test R2"],
                "External M0 test RMSE": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test RMSE"],
                "External M0 test MSE": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test MSE"],
                "External M0 test MAE": m0_z3cv7_reference["test_metrics"]["M0 Z3CV7 test MAE"],
            }

            internal_m0_model, internal_m0_scaler, internal_m0_search, internal_m0_train_pred, internal_m0_test_pred = train_rf_on_fixed_split(
                X_train_m3,
                y_train_m3,
                X_test_m3,
                cv,
            )
            internal_m0_train_metrics = calculate_metrics(y_train_m3, internal_m0_train_pred, "M3 base train")
            internal_m0_test_metrics = calculate_metrics(y_test_m3, internal_m0_test_pred, "M3 base test")

            # WR0 baseline.
            wr0_model, wr0_scaler, wr0_info, wr0_feature_cols, wr0_teacher_df = fit_ordinary_residual_model(
                teacher_clean,
                base_weather_features,
                internal_m0_model,
                internal_m0_scaler,
                sample_weight=None,
                task_suffix="WR0",
            )
            wr0_raw_train, wr0_raw_test, _, _ = make_ordinary_residual_predictions(
                wr0_model,
                wr0_scaler,
                wr0_feature_cols,
                train_df_m3,
                test_df_m3,
                base_weather_features,
            )
            wr0_rec, wr0_bundle = evaluate_m3(
                version="WR0_unweighted_residual_correction",
                y_train=y_train_m3,
                y_test=y_test_m3,
                internal_m0_train_pred=internal_m0_train_pred,
                internal_m0_test_pred=internal_m0_test_pred,
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
                student_clean=student_clean_m3,
                student_outliers=student_outliers,
                train_idx=train_idx,
                test_idx=test_idx,
                internal_m0_search=internal_m0_search,
                external_m0_search=external_m0_search,
                protein_info=protein_info,
                protein_oof_metrics=protein_oof_metrics,
                internal_m0_train_metrics=internal_m0_train_metrics,
                internal_m0_test_metrics=internal_m0_test_metrics,
                external_m0_train_metrics=external_m0_train_metrics,
                external_m0_test_metrics=external_m0_test_metrics,
                residual_info=wr0_info,
            )
            wr0_rec["WR0 M3 test R2"] = wr0_rec["M3 test R2"]
            wr0_rec["WR0 M3 test RMSE"] = wr0_rec["M3 test RMSE"]
            wr0_rec["WR0 M3 test MAE"] = wr0_rec["M3 test MAE"]
            wr0_rec["R5C M3 test R2"] = np.nan
            wr0_rec["R5C M3 test RMSE"] = np.nan
            wr0_rec["R5C M3 test MAE"] = np.nan
            wr0_rec["Delta test R2 M3_minus_WR0"] = 0.0
            wr0_rec["Delta test R2 M3_minus_R5C"] = np.nan
            wr0_rec = add_tmin_subset_metrics_to_record(wr0_rec, y_test_m3, wr0_bundle["m3_test_pred"], test_df_m3, prefix="M3")
            records.append(wr0_rec)

            # R5C baseline.
            r5_model, r5_info, r5_feature_cols, r5_teacher_df = fit_r5_conditional_residual_model(
                teacher_clean,
                base_weather_features,
                internal_m0_model,
                internal_m0_scaler,
                sample_weight=None,
                task_suffix="R5C",
            )
            r5_raw_train, r5_raw_test, _, _ = make_r5_residual_predictions(
                r5_model,
                r5_feature_cols,
                train_df_m3,
                test_df_m3,
                base_weather_features,
            )
            r5_rec, r5_bundle = evaluate_m3(
                version="R5C_conditional_constraint_no_weight",
                y_train=y_train_m3,
                y_test=y_test_m3,
                internal_m0_train_pred=internal_m0_train_pred,
                internal_m0_test_pred=internal_m0_test_pred,
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
                student_clean=student_clean_m3,
                student_outliers=student_outliers,
                train_idx=train_idx,
                test_idx=test_idx,
                internal_m0_search=internal_m0_search,
                external_m0_search=external_m0_search,
                protein_info=protein_info,
                protein_oof_metrics=protein_oof_metrics,
                internal_m0_train_metrics=internal_m0_train_metrics,
                internal_m0_test_metrics=internal_m0_test_metrics,
                external_m0_train_metrics=external_m0_train_metrics,
                external_m0_test_metrics=external_m0_test_metrics,
                residual_info=r5_info,
            )
            r5_rec["WR0 M3 test R2"] = wr0_rec["M3 test R2"]
            r5_rec["WR0 M3 test RMSE"] = wr0_rec["M3 test RMSE"]
            r5_rec["WR0 M3 test MAE"] = wr0_rec["M3 test MAE"]
            r5_rec["R5C M3 test R2"] = r5_rec["M3 test R2"]
            r5_rec["R5C M3 test RMSE"] = r5_rec["M3 test RMSE"]
            r5_rec["R5C M3 test MAE"] = r5_rec["M3 test MAE"]
            r5_rec["Delta test R2 M3_minus_WR0"] = r5_rec["M3 test R2"] - wr0_rec["M3 test R2"]
            r5_rec["Delta test R2 M3_minus_R5C"] = 0.0
            r5_rec = add_tmin_subset_metrics_to_record(r5_rec, y_test_m3, r5_bundle["m3_test_pred"], test_df_m3, prefix="M3")
            records.append(r5_rec)

            # R5CW grid.
            for spec in weighted_specs:
                print("  Training", spec["version"])
                residual_weight = None
                alpha_weight = None

                if spec["residual_weighted"]:
                    residual_weight = build_sample_weight(teacher_clean, scheme=spec["scheme"], beta=spec["beta"], protein_col=PROTEIN_COL)
                if spec["alpha_weighted"]:
                    alpha_weight = build_sample_weight(test_df_m3, scheme=spec["scheme"], beta=spec["beta"], protein_col=PREDICTED_PROTEIN_COL)

                cw_model, cw_info, cw_feature_cols, cw_teacher_df = fit_r5_conditional_residual_model(
                    teacher_clean,
                    base_weather_features,
                    internal_m0_model,
                    internal_m0_scaler,
                    sample_weight=residual_weight,
                    task_suffix=spec["version"],
                )
                cw_raw_train, cw_raw_test, _, _ = make_r5_residual_predictions(
                    cw_model,
                    cw_feature_cols,
                    train_df_m3,
                    test_df_m3,
                    base_weather_features,
                )
                alpha_metric = "weighted_r2" if spec["alpha_weighted"] else "r2"
                cw_rec, cw_bundle = evaluate_m3(
                    version=spec["version"],
                    y_train=y_train_m3,
                    y_test=y_test_m3,
                    internal_m0_train_pred=internal_m0_train_pred,
                    internal_m0_test_pred=internal_m0_test_pred,
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
                    student_clean=student_clean_m3,
                    student_outliers=student_outliers,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    internal_m0_search=internal_m0_search,
                    external_m0_search=external_m0_search,
                    protein_info=protein_info,
                    protein_oof_metrics=protein_oof_metrics,
                    internal_m0_train_metrics=internal_m0_train_metrics,
                    internal_m0_test_metrics=internal_m0_test_metrics,
                    external_m0_train_metrics=external_m0_train_metrics,
                    external_m0_test_metrics=external_m0_test_metrics,
                    weight_scheme=spec["scheme"],
                    beta=spec["beta"],
                    weight_location=spec["location"],
                    residual_weight=residual_weight,
                    alpha_weight=alpha_weight,
                    residual_info=cw_info,
                )
                cw_rec["WR0 M3 test R2"] = wr0_rec["M3 test R2"]
                cw_rec["WR0 M3 test RMSE"] = wr0_rec["M3 test RMSE"]
                cw_rec["WR0 M3 test MAE"] = wr0_rec["M3 test MAE"]
                cw_rec["R5C M3 test R2"] = r5_rec["M3 test R2"]
                cw_rec["R5C M3 test RMSE"] = r5_rec["M3 test RMSE"]
                cw_rec["R5C M3 test MAE"] = r5_rec["M3 test MAE"]
                cw_rec["Delta test R2 M3_minus_WR0"] = cw_rec["M3 test R2"] - wr0_rec["M3 test R2"]
                cw_rec["Delta test RMSE M3_minus_WR0"] = cw_rec["M3 test RMSE"] - wr0_rec["M3 test RMSE"]
                cw_rec["Delta test MAE M3_minus_WR0"] = cw_rec["M3 test MAE"] - wr0_rec["M3 test MAE"]
                cw_rec["Delta test R2 M3_minus_R5C"] = cw_rec["M3 test R2"] - r5_rec["M3 test R2"]
                cw_rec["Delta test RMSE M3_minus_R5C"] = cw_rec["M3 test RMSE"] - r5_rec["M3 test RMSE"]
                cw_rec["Delta test MAE M3_minus_R5C"] = cw_rec["M3 test MAE"] - r5_rec["M3 test MAE"]
                cw_rec["Improves over WR0 by R2"] = cw_rec["Delta test R2 M3_minus_WR0"] > 0
                cw_rec["Improves over R5C by R2"] = cw_rec["Delta test R2 M3_minus_R5C"] > 0
                cw_rec = add_tmin_subset_metrics_to_record(cw_rec, y_test_m3, cw_bundle["m3_test_pred"], test_df_m3, prefix="M3")
                records.append(cw_rec)

                is_better_weighted = best_weighted_record is None or (
                    cw_rec["M3 test R2"] > best_weighted_record["M3 test R2"] or
                    (np.isclose(cw_rec["M3 test R2"], best_weighted_record["M3 test R2"]) and cw_rec["M3 test RMSE"] < best_weighted_record["M3 test RMSE"])
                )

                if is_better_weighted:
                    best_weighted_record = cw_rec.copy()
                    best_weighted_bundle = {
                        "student_clean": student_clean_m3.copy(),
                        "train_df": train_df_m3.copy(),
                        "test_df": test_df_m3.copy(),
                        "external_m0_model": external_m0_model,
                        "external_m0_scaler": external_m0_scaler,
                        "external_m0_reference": m0_z3cv7_reference,
                        "external_m0_z3cv7_reference": m0_z3cv7_reference,
                        "internal_m0_model": internal_m0_model,
                        "internal_m0_scaler": internal_m0_scaler,
                        "internal_m0_train_pred": internal_m0_train_pred,
                        "internal_m0_test_pred": internal_m0_test_pred,
                        "residual_model": cw_model,
                        "residual_feature_cols": cw_feature_cols,
                        "raw_train_correction": cw_raw_train,
                        "raw_test_correction": cw_raw_test,
                        "alpha_info": cw_bundle["alpha_info"],
                        "alpha_df": cw_bundle["alpha_df"],
                        "train_alpha": cw_bundle["train_alpha"],
                        "test_alpha": cw_bundle["test_alpha"],
                        "m3_train_pred": cw_bundle["m3_train_pred"],
                        "m3_test_pred": cw_bundle["m3_test_pred"],
                        "teacher_residual_df": cw_teacher_df.copy(),
                        "y_train": y_train_m3,
                        "y_test": y_test_m3,
                        "base_weather_features": base_weather_features,
                    }

                is_better_any = best_any_record is None or (
                    cw_rec["M3 test R2"] > best_any_record["M3 test R2"] or
                    (np.isclose(cw_rec["M3 test R2"], best_any_record["M3 test R2"]) and cw_rec["M3 test RMSE"] < best_any_record["M3 test RMSE"])
                )
                if is_better_any:
                    best_any_record = cw_rec.copy()

    if len(records) == 0:
        raise RuntimeError("No valid model was trained.")

    result_df = pd.DataFrame(records)
    internal_output_cols = [
        c for c in result_df.columns
        if c.startswith("M3_base_model")
    ]
    if internal_output_cols:
        result_df = result_df.drop(columns=internal_output_cols)
    result_df.to_excel(os.path.join(out_dirs["tables"], "combined_residual_constraint_weighting_all_results.xlsx"), index=False)

    core_cols = [
        "Model family", "Version", "Weight scheme", "Weight beta", "Weight location",
        "z_score_threshold", "cv",
        "External M0 test R2", "External M0 test RMSE", "External M0 test MSE", "External M0 test MAE",
        "WR0 M3 test R2", "R5C M3 test R2",
        "M3 test R2", "M3 test RMSE", "M3 test MAE",
        "M3 Low TMIN test R2", "M3 Low TMIN test RMSE", "M3 Low TMIN test MAE",
        "M3 High TMIN test R2", "M3 High TMIN test RMSE", "M3 High TMIN test MAE",
        "Delta test R2 M3_minus_external_M0",
        "Delta test R2 M3_minus_WR0",
        "Delta test R2 M3_minus_R5C",
        "Improves over WR0 by R2", "Improves over R5C by R2",
        "External_M0_best_params",
        "External_M0_reference_train_R2", "External_M0_reference_test_R2",
        "External_M0_reference_abs_train_test_gap", "External_M0_reference_pass_overfit_filter",
        "Protein_proxy_cv_R2", "Protein_proxy_oof_R2",
        "Low group n", "High group n",
        "Residual train weight_min", "Residual train weight_max", "Residual train weight_mean", "Residual train weight_std",
        "Alpha test weight_min", "Alpha test weight_max", "Alpha test weight_mean", "Alpha test weight_std",
    ]
    core_cols_existing = [c for c in core_cols if c in result_df.columns]
    result_df[core_cols_existing].to_excel(os.path.join(out_dirs["tables"], "combined_residual_constraint_weighting_core_metrics.xlsx"), index=False)

    if best_weighted_record is not None:
        best_weighted_record_export = {k: v for k, v in best_weighted_record.items() if not (str(k).startswith("M3_base_model"))}
        pd.DataFrame([best_weighted_record_export]).to_excel(os.path.join(out_dirs["tables"], "best_R5CW_weighted_model_metrics.xlsx"), index=False)
    if best_any_record is not None:
        best_any_record_export = {k: v for k, v in best_any_record.items() if not (str(k).startswith("M3_base_model"))}
        pd.DataFrame([best_any_record_export]).to_excel(os.path.join(out_dirs["tables"], "best_any_model_metrics.xlsx"), index=False)

    if best_weighted_bundle is not None:
        joblib.dump(protein_model, os.path.join(out_dirs["models"], "protein_proxy_model.pkl"))
        joblib.dump(protein_scaler, os.path.join(out_dirs["models"], "protein_proxy_scaler.pkl"))
        joblib.dump(best_weighted_bundle["external_m0_model"], os.path.join(out_dirs["models"], "external_M0_model.pkl"))
        joblib.dump(best_weighted_bundle["external_m0_scaler"], os.path.join(out_dirs["models"], "external_M0_scaler.pkl"))
        joblib.dump(best_weighted_bundle["internal_m0_model"], os.path.join(out_dirs["models"], "best_M3_base_model.pkl"))
        joblib.dump(best_weighted_bundle["internal_m0_scaler"], os.path.join(out_dirs["models"], "best_M3_base_scaler.pkl"))
        joblib.dump(best_weighted_bundle["residual_model"], os.path.join(out_dirs["models"], "best_M3_conditional_residual_model.pkl"))
        joblib.dump(best_weighted_bundle["alpha_info"], os.path.join(out_dirs["models"], "best_M3_alpha_info.pkl"))

        train_pred_df = best_weighted_bundle["train_df"].copy()
        test_pred_df = best_weighted_bundle["test_df"].copy()

        train_pred_df["Raw residual correction"] = best_weighted_bundle["raw_train_correction"]
        train_pred_df["Adaptive alpha"] = best_weighted_bundle["train_alpha"]
        train_pred_df["M3 prediction"] = best_weighted_bundle["m3_train_pred"]
        train_pred_df["M3 residual"] = train_pred_df[original_target_col] - train_pred_df["M3 prediction"]

        test_pred_df["Raw residual correction"] = best_weighted_bundle["raw_test_correction"]
        test_pred_df["Adaptive alpha"] = best_weighted_bundle["test_alpha"]
        test_pred_df["M3 prediction"] = best_weighted_bundle["m3_test_pred"]
        test_pred_df["M3 residual"] = test_pred_df[original_target_col] - test_pred_df["M3 prediction"]

        train_pred_df.to_excel(os.path.join(out_dirs["tables"], "best_M3_train_predictions.xlsx"), index=False)
        test_pred_df.to_excel(os.path.join(out_dirs["tables"], "best_M3_test_predictions.xlsx"), index=False)

        metric_df = plot_independent_m0_z3cv7_vs_m3_metric_comparison(
            m0_z3cv7_reference,
            best_weighted_record,
            out_dirs["performance"],
        )
        independent_m0_m3_comparison_df = metric_df.copy()

        plot_true_vs_pred(m0_reference["y_train_best"], m0_reference["y_pred_train_best"],
                          m0_reference["y_test_best"], m0_reference["y_pred_test_best"],
                          "M0", out_dirs["performance"], "exact_M0_Z3CV7")
        plot_true_vs_pred(best_weighted_bundle["y_train"], best_weighted_bundle["m3_train_pred"],
                          best_weighted_bundle["y_test"], best_weighted_bundle["m3_test_pred"],
                          "M3", out_dirs["performance"], "best_M3")

        residual_diag_df = pd.DataFrame({
            "True": best_weighted_bundle["y_test"],
            "M3 prediction": best_weighted_bundle["m3_test_pred"],
        })
        residual_diag_df["M3 residual"] = residual_diag_df["True"] - residual_diag_df["M3 prediction"]
        residual_diag_df["M3 absolute residual"] = np.abs(residual_diag_df["M3 residual"])
        residual_diag_df.to_excel(os.path.join(out_dirs["performance"], "best_M3_test_residual_diagnostics.xlsx"), index=False)
        plt.figure(figsize=FIGSIZE["residual_scatter"])
        ax = sns.scatterplot(data=residual_diag_df, x="M3 prediction", y="M3 residual", s=120, alpha=0.7, edgecolor="k", color="#b4d4e1")
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Values", fontsize=18, fontweight="bold")
        ax.set_ylabel("Residuals", fontsize=18, fontweight="bold")
        ax.set_title("M3 Residual Plot", fontsize=20, fontweight="bold")
        format_axes_code2(ax, grid_axis="both")
        save_current_figure(os.path.join(out_dirs["performance"], "best_M3_test_residual_plot.png"))

        exact_m0_diag_df = pd.DataFrame({
            "True": m0_reference["y_test_best"],
            "M0 prediction": m0_reference["y_pred_test_best"],
            "M0 residual": m0_reference["y_test_best"] - m0_reference["y_pred_test_best"],
        })
        exact_m0_diag_df.to_excel(os.path.join(out_dirs["performance"], "exact_M0_Z3CV7_test_residual_diagnostics.xlsx"), index=False)

        high_tmin_df = evaluate_high_tmin_subset(best_weighted_bundle, out_dirs["high_tmin"])
        residual_prop_df = plot_residual_distribution_proportions(best_weighted_bundle, out_dirs["mechanism"])
        pdp_df = plot_constrained_pdp(best_weighted_bundle, out_dirs["mechanism"])
        alpha_validation_df = plot_alpha_rationality_validation(best_weighted_bundle, out_dirs["mechanism"])
        alpha_search_df = plot_alpha_search_diagnostics(best_weighted_bundle, out_dirs["mechanism"])
        importance_outputs = plot_importance_outputs(best_weighted_bundle, out_dirs["importance"])
        weight_df = plot_weight_outputs(best_weighted_record, teacher_clean, best_weighted_bundle["test_df"], out_dirs["learning"])
        beta_effect_df = plot_weight_beta_effects(result_df, out_dirs["learning"])
        scheme_beta_heatmap_outputs = plot_weight_scheme_beta_heatmaps(result_df, out_dirs["learning"])
        weight_tmin_effect_df = plot_weight_tmin_effects(result_df, out_dirs["learning"])
        screening_summary = plot_model_screening_diagnostics(result_df, out_dirs["learning"])

        # Use the same cleaned sample set as the selected fixed-test model.
        # This corrects the earlier underestimation of M3 in the training-fraction plot.
        fraction_df = run_training_fraction_sensitivity(
            aligned_m3_df=best_weighted_bundle["student_clean"],
            base_weather_features=base_weather_features,
            original_target_col=original_target_col,
            teacher_clean=teacher_clean,
            best_weighted_record=best_weighted_record,
            output_dir=out_dirs["training_fraction"],
            m0_quantity_df=m0_quantity_df,
        )

        summary_workbook = os.path.join(out_dirs["tables"], "M3_extended_outputs_summary.xlsx")
        with pd.ExcelWriter(summary_workbook) as writer:
            pd.DataFrame([best_weighted_record_export]).to_excel(writer, sheet_name="best_record", index=False)
            metric_df.to_excel(writer, sheet_name="metric_comparison", index=False)
            independent_m0_m3_comparison_df.to_excel(writer, sheet_name="independent_M0_M3", index=False)
            residual_diag_df.to_excel(writer, sheet_name="test_residuals", index=False)
            high_tmin_df.to_excel(writer, sheet_name="TMIN_subset_metrics", index=False)
            residual_prop_df.to_excel(writer, sheet_name="residual_proportion", index=False)
            if isinstance(pdp_df, pd.DataFrame):
                pdp_df.to_excel(writer, sheet_name="constrained_PDP", index=False)
            alpha_validation_df.to_excel(writer, sheet_name="alpha_validation", index=False)
            if isinstance(alpha_search_df, pd.DataFrame):
                alpha_search_df.to_excel(writer, sheet_name="alpha_search", index=False)
            fraction_df.to_excel(writer, sheet_name="training_fraction", index=False)
            if isinstance(screening_summary, pd.DataFrame):
                screening_summary.to_excel(writer, sheet_name="screening_summary", index=False)
            if isinstance(beta_effect_df, pd.DataFrame):
                beta_effect_df.to_excel(writer, sheet_name="beta_effect", index=False)
            if isinstance(weight_tmin_effect_df, pd.DataFrame):
                weight_tmin_effect_df.to_excel(writer, sheet_name="weight_TMIN_effect", index=False)
            for key, value in scheme_beta_heatmap_outputs.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=("wb_" + str(key))[:31], index=False)
            for key, value in importance_outputs.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=str(key)[:31], index=False)

        print("\nExtended outputs saved to classified folders under:", OUTPUT_DIR)
        print("Summary workbook:", summary_workbook)

    print("\n================ Combined residual + constraint + weighting grid search completed ================")
    print("Output directory:", OUTPUT_DIR)
    if best_weighted_record is not None:
        print("\nSelected M3 weighted model:")
        print("Version:", best_weighted_record["Version"])
        print("Best z-score:", best_weighted_record["z_score_threshold"])
        print("Best CV:", best_weighted_record["cv"])
        print("External M0 test R2:", best_weighted_record["External M0 test R2"])
        print("WR0 M3 test R2:", best_weighted_record["WR0 M3 test R2"])
        print("R5C M3 test R2:", best_weighted_record["R5C M3 test R2"])
        print("M3 test R2:", best_weighted_record["M3 test R2"])
        print("Delta test R2 M3_minus_external_M0:", best_weighted_record["Delta test R2 M3_minus_external_M0"])
        print("Delta test R2 M3_minus_WR0:", best_weighted_record["Delta test R2 M3_minus_WR0"])
        print("Delta test R2 M3_minus_R5C:", best_weighted_record["Delta test R2 M3_minus_R5C"])


if __name__ == "__main__":
    main()
