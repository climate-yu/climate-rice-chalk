# -*- coding: utf-8 -*-
"""
M2-W direct sample-weighted model on the M0 framework, without residual correction.
Expanded output version.

This script is modified from Code 1 by adding the main outputs used in Code 2:
1. Best-model final performance summary.
2. SHAP feature importance table.
3. SHAP beeswarm plot.
4. Custom SHAP bar plot.
5. SHAP dependence plot for the most important feature.
6. True vs predicted JointGrid plot.
7. Residual plot.
8. Learning curve.

Additional outputs specific to Code 1:
1. Best-model prediction table for train and test samples.
2. Best-model RF impurity feature importance.
3. Best-model permutation importance.
4. Sample-weight distribution plot.
5. Sample weight vs weight signal plot.
6. Residuals vs TMIN plot.
7. Residuals vs Protein_Heat_Risk plot.
8. Residual histogram.
9. Model comparison plot across the grid-search results.
10. Best weighted model vs corresponding M0 summary.

Important design rule:
Only the final best direct sample-weighted model receives detailed figures and model-result outputs.
The full grid-search table is still saved for traceability, but plots and interpretation tables are generated only for the best weighted model.
The selected best weighted model is explicitly exported as the best M2-W model, in addition to the generic best-weighted outputs.
"""

import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\Direct_M0_sample_weighting_only_gridsearch_results"
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

# These variables are used only for weight construction, never for model input features.
WEIGHT_ONLY_COLS = [TMIN_EXCESS_COL, TMIN_FLAG_COL, PROTEIN_RISK_COL, PREDICTED_PROTEIN_COL]

STUDENT_TEST_SIZE = 0.30
RANDOM_STATE = 42
TEACHER_Z_THRESHOLD = 4
STUDENT_Z_THRESHOLDS = [3]
STUDENT_CV_VALUES = [7]
TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10

PROTEIN_PROXY_N_ITER = 40
MODEL_N_ITER = 100

WEIGHT_SCHEMES = ["binary_tmin", "excess_tmin", "protein_heat_risk"]
WEIGHT_BETAS = [0.25, 0.5, 1.0, 2.0, 3.0]

# Output control.
TOP_K = 5
MODEL_DISPLAY_NAME = "M2-W"
OVERFITTING_GAP_LIMIT = None
# If you want to mimic Code 2's mild overfitting constraint, set OVERFITTING_GAP_LIMIT = 0.5.
# In this script, the default is None so that the best weighted model is selected strictly by Test R2 and Test RMSE.

# Unified plotting style, aligned with Code 2.
PLOT_PALETTE = {"Train": "#b4d4e1", "Test": "#f4ba8a"}
POINT_COLOR = "#b4d4e1"
TEST_COLOR = "#f4ba8a"
LOWESS_COLOR = "blue"
REFERENCE_COLOR = "black"
SHAP_DEPENDENCE_CI_HALF_WIDTH = 0.1

FIGSIZE = {
    "shap_dot": (6, 4),
    "shap_bar": (6, 4),
    "shap_dependence": (6, 5),
    "true_pred": (6, 6),
    "residuals": (6, 5),
    "residual_hist": (6, 5),
    "learning_curve": (6, 5),
    "rf_importance": (6, 4),
    "permutation_importance": (6, 4),
    "weight_distribution": (6, 5),
    "weight_signal": (6, 5),
    "residual_tmin": (6, 5),
    "residual_heat_risk": (6, 5),
    "model_comparison": (9, 5),
}
JOINTGRID_HEIGHT = 6

XAXIS_LABEL = {
    "shap_dot_x": "SHAP value",
    "shap_bar_x": "mean(|SHAP value|)",
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.size"] = 14


# =========================================================
# Parameter grids
# =========================================================
rf_param_grid = {
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


# =========================================================
# Utility functions
# =========================================================
def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def add_tmin_threshold_features(df):
    df = df.copy()
    if TMIN_COL not in df.columns:
        raise ValueError(f"Missing {TMIN_COL}. Cannot create TMIN threshold features.")
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


def build_weight_signal(df, scheme, protein_col=None):
    if scheme == "none":
        return np.zeros(len(df), dtype=float)
    if scheme == "binary_tmin":
        return (df[TMIN_COL].values > TMIN_THRESHOLD).astype(float)
    if scheme == "excess_tmin":
        return normalize_0_1(df[TMIN_EXCESS_COL].values)
    if scheme == "protein_heat_risk":
        if protein_col is None or protein_col not in df.columns:
            raise ValueError("protein_heat_risk weighting requires a valid protein_col.")
        return normalize_0_1(df[protein_col].values * df[TMIN_EXCESS_COL].values)
    raise ValueError(f"Unknown weight scheme: {scheme}")


def build_sample_weight(df, scheme, beta, protein_col=None):
    if scheme == "none" or beta == 0:
        return np.ones(len(df), dtype=float)
    signal = build_weight_signal(df, scheme, protein_col=protein_col)
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
    cleaned_df = df.drop(df.index[outliers]).copy()
    return cleaned_df, outliers


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_current_figure(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def tune_rf(df, feature_cols, target_col, param_grid, cv, n_iter, scoring="r2", sample_weight=None):
    X = df[feature_cols].values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
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
    return rs.best_estimator_, scaler, rs


def generate_oof_prediction_rf(df, feature_cols, target_col, best_params, n_splits=3):
    X = df[feature_cols].values
    y = df[target_col].values
    oof_pred = np.zeros(len(df), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, valid_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_valid = scaler.transform(X[valid_idx])
        model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
        model.fit(X_train, y[train_idx])
        oof_pred[valid_idx] = model.predict(X_valid)

    return oof_pred


def train_rf_on_fixed_split(X_train, y_train, X_test, cv, sample_weight=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=rf_param_grid,
        n_iter=MODEL_N_ITER,
        cv=cv,
        scoring="r2",
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)

    rs.fit(X_train_scaled, y_train, **fit_kwargs)
    model = rs.best_estimator_

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    return model, scaler, rs, train_pred, test_pred, X_train_scaled, X_test_scaled


# =========================================================
# Plot functions
# =========================================================
def plot_true_vs_pred(y_train, y_train_pred, y_test, y_test_pred, title, save_path):
    """True-vs-predicted JointGrid, harmonized with Code 2."""
    data_train = pd.DataFrame({"True": y_train, "Predicted": y_train_pred, "Data Set": "Train"})
    data_test = pd.DataFrame({"True": y_test, "Predicted": y_test_pred, "Data Set": "Test"})
    data_plot = pd.concat([data_train, data_test], axis=0)

    g = sns.JointGrid(
        data=data_plot,
        x="True",
        y="Predicted",
        hue="Data Set",
        height=JOINTGRID_HEIGHT,
        palette=PLOT_PALETTE,
    )
    g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
    sns.regplot(
        data=data_train,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color=PLOT_PALETTE["Train"],
        label="Train",
    )
    sns.regplot(
        data=data_test,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color=PLOT_PALETTE["Test"],
        label="Test",
    )
    g.plot_marginals(sns.histplot, kde=False, element="bars", multiple="stack", alpha=0.5)

    ax = g.ax_joint
    ax.set_xlabel("True Values", fontsize=20, weight="bold", labelpad=10)
    ax.set_ylabel("Predicted Values", fontsize=20, weight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = rmse_score(y_test, y_test_pred)
    ax.text(
        0.95,
        0.05,
        f"$R^2$ = {test_r2:.2f}\nRMSE = {test_rmse:.2f}",
        transform=ax.transAxes,
        fontsize=22,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )
    ax.text(
        0.75,
        0.99,
        MODEL_DISPLAY_NAME,
        transform=ax.transAxes,
        fontsize=18,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

    min_val = data_plot["True"].min()
    max_val = data_plot["True"].max()
    ax.plot([min_val, max_val], [min_val, max_val], c=REFERENCE_COLOR, linestyle="--", alpha=0.7, label="x=y")
    ax.legend(loc="best", fontsize=16)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

def plot_residuals(y_true, y_pred, save_path, title="Residual Plot"):
    residuals = y_true - y_pred
    plt.figure(figsize=FIGSIZE["residuals"])
    plt.scatter(y_pred, residuals, alpha=0.7, s=120, edgecolors="k", color=POINT_COLOR)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Predicted Values", fontsize=18)
    plt.ylabel("Residuals", fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    save_current_figure(save_path)

def plot_residual_histogram(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    plt.figure(figsize=FIGSIZE["residual_hist"])
    plt.hist(residuals, bins=15, edgecolor="black", alpha=0.75)
    plt.axvline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Residuals", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.title("Residual Distribution", fontsize=20)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_figure(save_path)


def plot_learning_curve(best_model, X_train_scaled, y_train, X_test_scaled, y_test, save_path):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        n = max(2, int(len(X_train_scaled) * train_size))
        X_train_subset = X_train_scaled[:n]
        y_train_subset = y_train[:n]

        model = RandomForestRegressor(**best_model.get_params())
        model.fit(X_train_subset, y_train_subset)

        train_scores.append(r2_score(y_train_subset, model.predict(X_train_subset)))
        test_scores.append(r2_score(y_test, model.predict(X_test_scaled)))

    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    plt.figure(figsize=FIGSIZE["learning_curve"])
    plt.plot(train_sizes * len(X_train_scaled), train_scores, "o-", color="r", label="Training score", linewidth=2)
    plt.plot(train_sizes * len(X_train_scaled), test_scores, "o-", color="b", label="Testing score", linewidth=2)
    plt.xlabel("Training Set Size", fontsize=18)
    plt.ylabel("Score (R²)", fontsize=18)
    plt.title("Learning Curve", fontsize=20)
    plt.legend(loc="best", fontsize=14)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1)
    save_current_figure(save_path)

def plot_barh(values, names, save_path, figsize, xlabel, title=None):
    values = np.asarray(values, dtype=float)
    names = np.asarray(names, dtype=object)

    order = np.argsort(values)[::-1]
    values_sorted = values[order]
    names_sorted = names[order]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(values_sorted))
    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(values_sorted)))
    ax.barh(
        y,
        values_sorted[::-1],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        height=0.7,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted[::-1], fontsize=14, fontweight="bold")

    max_val = np.nanmax(values_sorted) if len(values_sorted) > 0 else 0.0
    offset = max(max_val * 0.01, 1e-9)
    for yi, vi in zip(y, values_sorted[::-1]):
        ax.text(vi + offset, yi, f"{vi:.3f}", va="center", ha="left", fontsize=12)

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    save_current_figure(save_path)

def plot_shap_bar_custom(values, names, save_path, figsize=(6, 4), xlabel="mean(|SHAP value|)"):
    plot_barh(
        values=values,
        names=names,
        save_path=save_path,
        figsize=figsize,
        xlabel=xlabel,
        title="SHAP Feature Importance",
    )


def plot_shap_outputs(model, X_test_scaled, scaler, feature_cols, output_dir):
    if not SHAP_AVAILABLE:
        print("SHAP is not available. SHAP outputs were skipped.")
        return None

    shap_dir = ensure_dir(os.path.join(output_dir, "best_model_SHAP_outputs"))

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Some SHAP versions return a list for multi-output models. This script expects single-output regression.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_mean_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP Mean Importance": shap_mean_importance,
    }).sort_values(by="SHAP Mean Importance", ascending=False)

    feature_importance_df.to_excel(
        os.path.join(shap_dir, "best_model_shap_feature_importances.xlsx"),
        index=False,
    )

    topk_df = feature_importance_df.head(min(TOP_K, len(feature_importance_df))).copy()
    topk_names = topk_df["Feature"].values
    topk_idx = [feature_cols.index(name) for name in topk_names]

    # SHAP beeswarm
    shap.summary_plot(
        shap_values[:, topk_idx],
        X_test_scaled[:, topk_idx],
        feature_names=topk_names,
        plot_type="dot",
        show=False,
        plot_size=FIGSIZE["shap_dot"],
    )
    ax = plt.gca()
    ax.set_xlabel(XAXIS_LABEL["shap_dot_x"], fontsize=14, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "best_model_shap_beeswarm_top_features.png"), dpi=600, bbox_inches="tight")
    plt.close()

    # SHAP bar
    plot_shap_bar_custom(
        values=topk_df["SHAP Mean Importance"].values,
        names=topk_df["Feature"].values,
        save_path=os.path.join(shap_dir, "best_model_shap_bar_top_features.png"),
        figsize=FIGSIZE["shap_bar"],
        xlabel=XAXIS_LABEL["shap_bar_x"],
    )

    # SHAP dependence plot for the most important feature, aligned with Code 2.
    best_feature = topk_df.iloc[0]["Feature"]
    feature_index = feature_cols.index(best_feature)
    shap_values_for_feature = shap_values[:, feature_index]
    original_feature_values = scaler.inverse_transform(X_test_scaled)[:, feature_index]

    plt.figure(figsize=FIGSIZE["shap_dependence"])
    if STATSMODELS_AVAILABLE and len(np.unique(original_feature_values)) >= 3:
        lowess_fit = sm.nonparametric.lowess(shap_values_for_feature, original_feature_values, frac=0.7)
        plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color=LOWESS_COLOR, linewidth=3, label="Lowess")
        plt.fill_between(
            lowess_fit[:, 0],
            lowess_fit[:, 1] - SHAP_DEPENDENCE_CI_HALF_WIDTH,
            lowess_fit[:, 1] + SHAP_DEPENDENCE_CI_HALF_WIDTH,
            color="gray",
            alpha=0.3,
            label="CI",
        )
        plt.legend(fontsize=14)
    else:
        plt.scatter(original_feature_values, shap_values_for_feature, s=70, alpha=0.7, edgecolors="k", color=POINT_COLOR)

    plt.grid(True, linestyle="--", linewidth=1, color="lightgray", alpha=0.7)
    plt.xlabel(best_feature, fontsize=18)
    plt.ylabel("SHAP Value", fontsize=18)
    plt.title("SHAP Dependence Plot", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=14)
    save_current_figure(os.path.join(shap_dir, f"best_model_shap_dependence_{best_feature}.png"))

    return {
        "shap_values": shap_values,
        "feature_importance_df": feature_importance_df,
        "best_feature": best_feature,
        "shap_dir": shap_dir,
    }


def plot_rf_feature_importance(model, feature_cols, output_dir):
    importance = np.asarray(model.feature_importances_, dtype=float)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "RF impurity importance": importance,
    }).sort_values("RF impurity importance", ascending=False)

    importance_df.to_excel(
        os.path.join(output_dir, "best_model_rf_impurity_feature_importance.xlsx"),
        index=False,
    )

    top_df = importance_df.head(min(TOP_K, len(importance_df))).copy()
    plot_barh(
        values=top_df["RF impurity importance"].values,
        names=top_df["Feature"].values,
        save_path=os.path.join(output_dir, "best_model_rf_impurity_feature_importance_top_features.png"),
        figsize=FIGSIZE["rf_importance"],
        xlabel="RF impurity importance",
        title="RF Feature Importance",
    )

    return importance_df


def plot_permutation_importance(model, X_test_scaled, y_test, feature_cols, output_dir):
    perm = permutation_importance(
        model,
        X_test_scaled,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "Feature": feature_cols,
        "Permutation importance mean": perm.importances_mean,
        "Permutation importance std": perm.importances_std,
    }).sort_values("Permutation importance mean", ascending=False)

    perm_df.to_excel(
        os.path.join(output_dir, "best_model_permutation_importance.xlsx"),
        index=False,
    )

    top_df = perm_df.head(min(TOP_K, len(perm_df))).copy()
    plot_barh(
        values=top_df["Permutation importance mean"].values,
        names=top_df["Feature"].values,
        save_path=os.path.join(output_dir, "best_model_permutation_importance_top_features.png"),
        figsize=FIGSIZE["permutation_importance"],
        xlabel="Permutation importance mean decrease in R²",
        title="Permutation Feature Importance",
    )

    return perm_df


def plot_weight_distribution(train_weight, test_weight, output_dir):
    plt.figure(figsize=FIGSIZE["weight_distribution"])
    plt.hist(train_weight, bins=15, alpha=0.65, edgecolor="black", label="Train")
    plt.hist(test_weight, bins=15, alpha=0.65, edgecolor="black", label="Test")
    plt.xlabel("Sample weight", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.title("Sample Weight Distribution", fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_figure(os.path.join(output_dir, "best_model_sample_weight_distribution.png"))


def plot_weight_signal(test_df, test_weight, scheme, beta, output_dir):
    if scheme == "none":
        return

    signal = build_weight_signal(test_df, scheme=scheme, protein_col=PREDICTED_PROTEIN_COL)

    plt.figure(figsize=FIGSIZE["weight_signal"])
    plt.scatter(signal, test_weight, s=100, alpha=0.75, edgecolors="k", color=POINT_COLOR)
    plt.xlabel("Weight signal", fontsize=18)
    plt.ylabel("Sample weight", fontsize=18)
    plt.title("Sample Weight", fontsize=20)
    plt.grid(alpha=0.3, linestyle="--")
    save_current_figure(os.path.join(output_dir, "best_model_weight_signal_vs_sample_weight.png"))


def plot_residual_against_variable(test_df, y_test, y_test_pred, variable, output_dir):
    if variable not in test_df.columns:
        return

    residuals = y_test - y_test_pred
    plt.figure(figsize=FIGSIZE["residual_tmin"] if variable == TMIN_COL else FIGSIZE["residual_heat_risk"])
    plt.scatter(test_df[variable].values, residuals, s=100, alpha=0.75, edgecolors="k", color=POINT_COLOR)
    plt.axhline(0, color="r", linestyle="--", linewidth=2)

    if STATSMODELS_AVAILABLE and len(np.unique(test_df[variable].values)) >= 3:
        lowess_fit = sm.nonparametric.lowess(residuals, test_df[variable].values, frac=0.7)
        plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="blue", linewidth=3, label="Lowess Fit")
        plt.legend(fontsize=14)

    plt.xlabel(variable, fontsize=18)
    plt.ylabel("Residuals", fontsize=18)
    plt.title(f"Residuals vs {variable}", fontsize=20)
    plt.grid(alpha=0.3, linestyle="--")

    safe_var = variable.replace("/", "_").replace("\\", "_").replace(" ", "_")
    save_current_figure(os.path.join(output_dir, f"best_model_residuals_vs_{safe_var}.png"))


def plot_model_comparison(result_df, output_dir):
    weighted_df = result_df[result_df["Version"] != "M0_RF_unweighted_baseline"].copy()
    if weighted_df.empty:
        return

    plot_df = weighted_df.sort_values("Test R2", ascending=False).head(20).copy()
    plot_df["Model label"] = (
        plot_df["Version"].astype(str)
        + "\nZ="
        + plot_df["z_score_threshold"].astype(str)
        + ", CV="
        + plot_df["cv"].astype(str)
    )

    plt.figure(figsize=FIGSIZE["model_comparison"])
    y = np.arange(len(plot_df))
    plt.barh(y, plot_df["Test R2"].values[::-1], edgecolor="black", alpha=0.8)
    plt.yticks(y, plot_df["Model label"].values[::-1], fontsize=9)
    plt.xlabel("Test R²", fontsize=16)
    plt.ylabel("Weighted model configuration", fontsize=16)
    plt.title("Top Weighted Models by Test R²", fontsize=18)
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    save_current_figure(os.path.join(output_dir, "top_weighted_models_test_r2_comparison.png"))


def save_best_predictions(best_bundle, feature_cols, original_target_col, output_dir):
    train_df = best_bundle["train_df"].copy()
    test_df = best_bundle["test_df"].copy()

    train_pred = best_bundle["train_pred"]
    test_pred = best_bundle["test_pred"]
    y_train = best_bundle["y_train"]
    y_test = best_bundle["y_test"]

    train_out = train_df.copy()
    train_out["Data Set"] = "Train"
    train_out["Observed"] = y_train
    train_out["Predicted"] = train_pred
    train_out["Residual"] = y_train - train_pred
    train_out["Absolute error"] = np.abs(y_train - train_pred)
    train_out["Sample weight"] = best_bundle["train_weight"]

    test_out = test_df.copy()
    test_out["Data Set"] = "Test"
    test_out["Observed"] = y_test
    test_out["Predicted"] = test_pred
    test_out["Residual"] = y_test - test_pred
    test_out["Absolute error"] = np.abs(y_test - test_pred)
    test_out["Sample weight"] = best_bundle["test_weight"]

    pred_df = pd.concat([train_out, test_out], axis=0)
    pred_df.to_excel(
        os.path.join(output_dir, "best_model_train_test_predictions.xlsx"),
        index=False,
    )

    return pred_df


def save_final_metrics_text(best_weighted, best_m0, best_bundle, output_dir):
    y_train = best_bundle["y_train"]
    y_test = best_bundle["y_test"]
    train_pred = best_bundle["train_pred"]
    test_pred = best_bundle["test_pred"]

    final_metrics = {}
    final_metrics.update(calculate_metrics(y_train, train_pred, "Training"))
    final_metrics.update(calculate_metrics(y_test, test_pred, "Testing"))
    final_metrics.update({
        "Best Version": best_weighted["Version"],
        "Best Weight scheme": best_weighted["Weight scheme"],
        "Best Weight beta": best_weighted["Weight beta"],
        "Best z_score_threshold": best_weighted["z_score_threshold"],
        "Best cv": best_weighted["cv"],
        "Best params": best_weighted["Best params"],
        "M0 Test R2 within same Z-CV": best_weighted["M0 Test R2 within same Z-CV"],
        "Delta test R2 vs M0": best_weighted["Delta test R2 vs M0"],
        "Delta test RMSE vs M0": best_weighted["Delta test RMSE vs M0"],
        "Delta test MAE vs M0": best_weighted["Delta test MAE vs M0"],
        "Best M0 global Test R2": best_m0["Test R2"],
        "Best M0 global Test RMSE": best_m0["Test RMSE"],
        "Best M0 global Test MAE": best_m0["Test MAE"],
    })

    pd.DataFrame([final_metrics]).to_excel(
        os.path.join(output_dir, "best_model_final_performance_summary.xlsx"),
        index=False,
    )

    with open(os.path.join(output_dir, "best_model_final_performance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, "best_model_final_performance_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Final Model Performance\n")
        f.write("=======================\n")
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")

    return final_metrics


# =========================================================
# Main workflow
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

    if TMIN_COL not in base_weather_features:
        raise ValueError(f"{TMIN_COL} must be in base weather features.")

    print("\nM0 and weighted model input features, excluding weight-only variables:")
    print(base_weather_features)
    print("\nWeight-only variables:")
    print([TMIN_EXCESS_COL, TMIN_FLAG_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL])

    pd.DataFrame({"model_input_features": base_weather_features}).to_excel(
        os.path.join(OUTPUT_DIR, "direct_weighting_model_input_features.xlsx"),
        index=False,
    )

    # Teacher data for protein proxy only.
    teacher_df = teacher_numeric[
        [TARGET_COL, PROTEIN_COL] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]
    ].dropna().copy()

    teacher_clean, teacher_outliers = remove_outliers_by_zscore(
        teacher_df,
        base_weather_features + [PROTEIN_COL],
        TEACHER_Z_THRESHOLD,
    )

    print("\nTeacher valid n:", len(teacher_df))
    print("Teacher cleaned n:", len(teacher_clean))
    print("Teacher outliers removed:", len(teacher_outliers))

    teacher_cv = RepeatedKFold(
        n_splits=TEACHER_N_SPLITS,
        n_repeats=TEACHER_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    protein_model, protein_scaler, protein_search = tune_rf(
        teacher_clean,
        base_weather_features,
        PROTEIN_COL,
        protein_proxy_param_grid,
        teacher_cv,
        PROTEIN_PROXY_N_ITER,
        scoring="r2",
    )

    oof_protein = generate_oof_prediction_rf(
        teacher_clean,
        base_weather_features,
        PROTEIN_COL,
        protein_search.best_params_,
        n_splits=TEACHER_N_SPLITS,
    )

    protein_oof_metrics = calculate_metrics(
        teacher_clean[PROTEIN_COL].values,
        oof_protein,
        "Protein OOF prediction",
    )

    print("\nProtein proxy best params:", protein_search.best_params_)
    print("Protein proxy CV R2:", protein_search.best_score_)
    print("Protein proxy OOF metrics:", protein_oof_metrics)

    pd.DataFrame([{
        "Protein proxy best params": json.dumps(protein_search.best_params_, ensure_ascii=False),
        "Protein proxy CV R2": float(protein_search.best_score_),
        **protein_oof_metrics,
    }]).to_excel(
        os.path.join(OUTPUT_DIR, "protein_proxy_model_metrics.xlsx"),
        index=False,
    )

    # Student data with weight-only knowledge features.
    student_df = original_numeric[
        [original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]
    ].dropna().copy()

    student_df[PREDICTED_PROTEIN_COL] = protein_model.predict(
        protein_scaler.transform(student_df[base_weather_features].values)
    )
    student_df = add_protein_heat_risk(student_df, PREDICTED_PROTEIN_COL)

    print("\nStudent valid n:", len(student_df))

    specs = [{
        "Version": "M0_RF_unweighted_baseline",
        "Description": "Original M0 RF baseline. No sample weighting.",
        "weight_scheme": "none",
        "beta": 0.0,
    }]

    for scheme in WEIGHT_SCHEMES:
        for beta in WEIGHT_BETAS:
            specs.append({
                "Version": f"W_{scheme}_beta{str(beta).replace('.', '_')}",
                "Description": (
                    f"Direct RF model using sample weights from {scheme}, beta={beta}. "
                    "No residual correction and no added knowledge features."
                ),
                "weight_scheme": scheme,
                "beta": beta,
            })

    records = []
    bundles = {}

    for z_thr in STUDENT_Z_THRESHOLDS:
        student_clean, student_outliers = remove_outliers_by_zscore(student_df, base_weather_features, z_thr)

        if len(student_clean) < 30:
            print(f"Z-score={z_thr}: cleaned sample size too small; skipped.")
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

        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train = train_df[base_weather_features].values
        X_test = test_df[base_weather_features].values

        for cv_value in STUDENT_CV_VALUES:
            if cv_value >= len(X_train):
                continue

            print("\n" + "=" * 80)
            print(f"Training direct sample-weighting models: Z-score={z_thr}, CV={cv_value}")

            m0_test_r2_current = None
            m0_test_rmse_current = None
            m0_test_mae_current = None
            combo_records_start = len(records)

            for spec in specs:
                print("Training", spec["Version"])

                if spec["weight_scheme"] == "none":
                    train_weight = np.ones(len(train_df), dtype=float)
                    test_weight = np.ones(len(test_df), dtype=float)
                    fit_train_weight = None
                else:
                    train_weight = build_sample_weight(
                        train_df,
                        scheme=spec["weight_scheme"],
                        beta=spec["beta"],
                        protein_col=PREDICTED_PROTEIN_COL,
                    )
                    test_weight = build_sample_weight(
                        test_df,
                        scheme=spec["weight_scheme"],
                        beta=spec["beta"],
                        protein_col=PREDICTED_PROTEIN_COL,
                    )
                    fit_train_weight = train_weight

                model, scaler, search, train_pred, test_pred, X_train_scaled, X_test_scaled = train_rf_on_fixed_split(
                    X_train,
                    y_train,
                    X_test,
                    cv=cv_value,
                    sample_weight=fit_train_weight,
                )

                rec = {
                    "Version": spec["Version"],
                    "Description": spec["Description"],
                    "Model type": "RandomForestRegressor",
                    "Weight scheme": spec["weight_scheme"],
                    "Weight beta": spec["beta"],
                    "Feature cols": str(base_weather_features),
                    "Best params": json.dumps(search.best_params_, ensure_ascii=False),
                    "CV R2 on train": float(search.best_score_),
                    "z_score_threshold": z_thr,
                    "cv": cv_value,
                    "n_samples_after_cleaning": len(student_clean),
                    "n_outliers_removed": len(student_outliers),
                    "outlier_removed_ratio": float(len(student_outliers) / len(student_df)),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "Protein proxy CV R2": float(protein_search.best_score_),
                    "Protein proxy OOF R2": protein_oof_metrics["Protein OOF prediction R2"],
                    "Protein proxy OOF RMSE": protein_oof_metrics["Protein OOF prediction RMSE"],
                    "Protein proxy OOF MSE": protein_oof_metrics["Protein OOF prediction MSE"],
                    "Protein proxy OOF MAE": protein_oof_metrics["Protein OOF prediction MAE"],
                }

                rec.update({f"Train {k}": v for k, v in summarize_weight(train_weight).items()})
                rec.update({f"Test {k}": v for k, v in summarize_weight(test_weight).items()})

                rec.update(calculate_metrics(y_train, train_pred, "Train"))
                rec.update(calculate_metrics(y_test, test_pred, "Test"))

                rec.update({
                    "Weighted test R2": weighted_r2(y_test, test_pred, test_weight),
                    "Weighted test RMSE": weighted_rmse(y_test, test_pred, test_weight),
                    "Weighted test MAE": weighted_mae(y_test, test_pred, test_weight),
                    "Train-Test R2 gap": rec["Train R2"] - rec["Test R2"],
                    "Abs Train-Test R2 gap": abs(rec["Train R2"] - rec["Test R2"]),
                })

                records.append(rec)

                bundle_key = f"Z{z_thr}_CV{cv_value}_{spec['Version']}"
                bundles[bundle_key] = {
                    "model": model,
                    "scaler": scaler,
                    "search": search,
                    "train_pred": train_pred,
                    "test_pred": test_pred,
                    "train_weight": train_weight,
                    "test_weight": test_weight,
                    "fit_train_weight": fit_train_weight,
                    "y_train": y_train,
                    "y_test": y_test,
                    "X_train_scaled": X_train_scaled,
                    "X_test_scaled": X_test_scaled,
                    "train_df": train_df.copy(),
                    "test_df": test_df.copy(),
                    "z_thr": z_thr,
                    "cv": cv_value,
                    "version": spec["Version"],
                    "weight_scheme": spec["weight_scheme"],
                    "weight_beta": spec["beta"],
                }

                if spec["Version"] == "M0_RF_unweighted_baseline":
                    m0_test_r2_current = rec["Test R2"]
                    m0_test_rmse_current = rec["Test RMSE"]
                    m0_test_mae_current = rec["Test MAE"]

            # Add deltas within the same Z-score and CV combination.
            for j in range(combo_records_start, len(records)):
                records[j]["M0 Test R2 within same Z-CV"] = m0_test_r2_current
                records[j]["M0 Test RMSE within same Z-CV"] = m0_test_rmse_current
                records[j]["M0 Test MAE within same Z-CV"] = m0_test_mae_current
                records[j]["Delta test R2 vs M0"] = records[j]["Test R2"] - m0_test_r2_current
                records[j]["Delta test RMSE vs M0"] = records[j]["Test RMSE"] - m0_test_rmse_current
                records[j]["Delta test MAE vs M0"] = records[j]["Test MAE"] - m0_test_mae_current

    result_df = pd.DataFrame(records)

    if result_df.empty:
        raise RuntimeError("No valid direct sample-weighting results were generated.")

    result_df.to_excel(
        os.path.join(OUTPUT_DIR, "direct_sample_weighting_only_all_results_gridsearch.xlsx"),
        index=False,
    )

    core_cols = [
        "z_score_threshold", "cv", "Version", "Description", "Weight scheme", "Weight beta",
        "Test R2", "Test RMSE", "Test MSE", "Test MAE",
        "M0 Test R2 within same Z-CV", "M0 Test RMSE within same Z-CV", "M0 Test MAE within same Z-CV",
        "Delta test R2 vs M0", "Delta test RMSE vs M0", "Delta test MAE vs M0",
        "Weighted test R2", "Weighted test RMSE", "Weighted test MAE",
        "Train R2", "Train RMSE", "Train MSE", "Train MAE", "Train-Test R2 gap",
        "CV R2 on train", "Best params", "Feature cols",
    ]

    available_core_cols = [c for c in core_cols if c in result_df.columns]
    result_df[available_core_cols].to_excel(
        os.path.join(OUTPUT_DIR, "direct_sample_weighting_only_core_metrics_gridsearch.xlsx"),
        index=False,
    )

    plot_model_comparison(result_df, OUTPUT_DIR)

    # Select best weighted model only.
    weighted_df = result_df[result_df["Version"] != "M0_RF_unweighted_baseline"].copy()

    if OVERFITTING_GAP_LIMIT is not None:
        constrained = weighted_df[weighted_df["Abs Train-Test R2 gap"] <= OVERFITTING_GAP_LIMIT].copy()
        if not constrained.empty:
            weighted_df = constrained

    best_weighted = weighted_df.sort_values(["Test R2", "Test RMSE"], ascending=[False, True]).iloc[0].to_dict()

    pd.DataFrame([best_weighted]).to_excel(
        os.path.join(OUTPUT_DIR, "best_direct_sample_weighting_model_metrics_gridsearch.xlsx"),
        index=False,
    )

    # Explicit M2-W output. This is the same selected weighted model,
    # exported with M2-W naming for direct reporting and traceability.
    pd.DataFrame([best_weighted]).to_excel(
        os.path.join(OUTPUT_DIR, "best_M2W_model_metrics_gridsearch.xlsx"),
        index=False,
    )

    # Best M0 across all Z-score and CV combinations.
    m0_df = result_df[result_df["Version"] == "M0_RF_unweighted_baseline"].copy()
    best_m0 = m0_df.sort_values(["Test R2", "Test RMSE"], ascending=[False, True]).iloc[0].to_dict()

    pd.DataFrame([best_m0]).to_excel(
        os.path.join(OUTPUT_DIR, "best_M0_baseline_metrics_gridsearch.xlsx"),
        index=False,
    )

    # Save protein proxy model.
    joblib.dump(protein_model, os.path.join(OUTPUT_DIR, "protein_proxy_model.pkl"))
    joblib.dump(protein_scaler, os.path.join(OUTPUT_DIR, "protein_proxy_scaler.pkl"))

    best_weighted_key = f"Z{best_weighted['z_score_threshold']}_CV{best_weighted['cv']}_{best_weighted['Version']}"
    best_m0_key = f"Z{best_m0['z_score_threshold']}_CV{best_m0['cv']}_{best_m0['Version']}"

    if best_weighted_key not in bundles:
        raise KeyError(f"Best weighted bundle not found: {best_weighted_key}")

    best_bundle = bundles[best_weighted_key]

    # Save only best weighted model.
    joblib.dump(best_bundle["model"], os.path.join(OUTPUT_DIR, "best_direct_sample_weighting_model.pkl"))
    joblib.dump(best_bundle["scaler"], os.path.join(OUTPUT_DIR, "best_direct_sample_weighting_scaler.pkl"))

    # Explicit M2-W model objects. The selected best weighted model is the final M2-W model.
    joblib.dump(best_bundle["model"], os.path.join(OUTPUT_DIR, "best_M2W_model.pkl"))
    joblib.dump(best_bundle["scaler"], os.path.join(OUTPUT_DIR, "best_M2W_scaler.pkl"))

    # Save M0 model only as a reference object, without detailed model-result figures.
    if best_m0_key in bundles:
        joblib.dump(bundles[best_m0_key]["model"], os.path.join(OUTPUT_DIR, "best_M0_RF_baseline_model.pkl"))
        joblib.dump(bundles[best_m0_key]["scaler"], os.path.join(OUTPUT_DIR, "best_M0_RF_baseline_scaler.pkl"))

    # Detailed outputs for the best weighted model only.
    best_output_dir = ensure_dir(os.path.join(OUTPUT_DIR, "BEST_WEIGHTED_MODEL_ONLY_OUTPUTS"))

    save_best_predictions(best_bundle, base_weather_features, original_target_col, best_output_dir)
    final_metrics = save_final_metrics_text(best_weighted, best_m0, best_bundle, best_output_dir)

    y_train = best_bundle["y_train"]
    y_test = best_bundle["y_test"]
    train_pred = best_bundle["train_pred"]
    test_pred = best_bundle["test_pred"]
    X_train_scaled = best_bundle["X_train_scaled"]
    X_test_scaled = best_bundle["X_test_scaled"]
    model = best_bundle["model"]
    scaler = best_bundle["scaler"]

    plot_true_vs_pred(
        y_train,
        train_pred,
        y_test,
        test_pred,
        title=None,
        save_path=os.path.join(best_output_dir, "best_model_true_vs_predicted.png"),
    )

    plot_residuals(
        y_test,
        test_pred,
        save_path=os.path.join(best_output_dir, "best_model_test_residual_plot.png"),
        title="Residual Plot",
    )

    plot_residual_histogram(
        y_test,
        test_pred,
        save_path=os.path.join(best_output_dir, "best_model_test_residual_distribution.png"),
    )

    plot_learning_curve(
        model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        save_path=os.path.join(best_output_dir, "best_model_learning_curve.png"),
    )

    plot_rf_feature_importance(model, base_weather_features, best_output_dir)
    plot_permutation_importance(model, X_test_scaled, y_test, base_weather_features, best_output_dir)

    plot_weight_distribution(
        best_bundle["train_weight"],
        best_bundle["test_weight"],
        best_output_dir,
    )

    plot_weight_signal(
        best_bundle["test_df"],
        best_bundle["test_weight"],
        scheme=best_bundle["weight_scheme"],
        beta=best_bundle["weight_beta"],
        output_dir=best_output_dir,
    )

    plot_residual_against_variable(
        best_bundle["test_df"],
        y_test,
        test_pred,
        variable=TMIN_COL,
        output_dir=best_output_dir,
    )

    plot_residual_against_variable(
        best_bundle["test_df"],
        y_test,
        test_pred,
        variable=PROTEIN_RISK_COL,
        output_dir=best_output_dir,
    )

    shap_result = plot_shap_outputs(
        model,
        X_test_scaled,
        scaler,
        base_weather_features,
        best_output_dir,
    )

    if shap_result is not None:
        print("\nMost important SHAP feature:", shap_result["best_feature"])
        print("\nSHAP feature importance table:")
        print(shap_result["feature_importance_df"])

    # Explicit M2-W detailed-output directory. The contents are copied from the
    # generic best-weighted directory so that all figures and tables are also
    # available under the final model name. This does not retrain or reselect models.
    m2w_output_dir = ensure_dir(os.path.join(OUTPUT_DIR, "BEST_M2W_MODEL_OUTPUTS"))
    shutil.copytree(best_output_dir, m2w_output_dir, dirs_exist_ok=True)

    # Additional explicit M2-W summary files at the root output directory.
    pd.DataFrame([final_metrics]).to_excel(
        os.path.join(OUTPUT_DIR, "best_M2W_final_performance_summary.xlsx"),
        index=False,
    )
    with open(os.path.join(OUTPUT_DIR, "best_M2W_final_performance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=4)

    # Console output aligned with Code 2.
    print("\n================ Direct sample-weighting-only grid-search completed ================")
    print("Output directory:", OUTPUT_DIR)
    print("Best weighted model detailed output directory:", best_output_dir)
    print("Best M2-W model detailed output directory:", m2w_output_dir)

    print("\nBest M0 baseline:")
    print({k: best_m0[k] for k in ["z_score_threshold", "cv", "Test R2", "Test RMSE", "Test MSE", "Test MAE"]})

    print("\nBest weighted model:")
    print({
        k: best_weighted[k]
        for k in [
            "Version",
            "z_score_threshold",
            "cv",
            "Weight scheme",
            "Weight beta",
            "Test R2",
            "Test RMSE",
            "Test MSE",
            "Test MAE",
            "Delta test R2 vs M0",
        ]
    })

    print("\nFinal Model Performance:")
    print(f"Training R²: {final_metrics['Training R2']:.4f}")
    print(f"Testing  R²: {final_metrics['Testing R2']:.4f}")
    print(f"Training RMSE: {final_metrics['Training RMSE']:.4f}")
    print(f"Testing  RMSE: {final_metrics['Testing RMSE']:.4f}")
    print(f"Training  MSE: {final_metrics['Training MSE']:.4f}")
    print(f"Testing   MSE: {final_metrics['Testing MSE']:.4f}")
    print(f"Training  MAE: {final_metrics['Training MAE']:.4f}")
    print(f"Testing   MAE: {final_metrics['Testing MAE']:.4f}")


if __name__ == "__main__":
    main()
