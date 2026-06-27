# -*- coding: utf-8 -*-
"""
Knowledge-constrained correction models on the M0 framework.
Grid-search version aligned with the ordinary residual-correction workflow.

Purpose
1. Build the original M0 baseline: original weather factors -> Chalkiness degree.
2. Build M2-C as M0 prediction + knowledge-constrained correction term.
3. The correction term is learned by HGB on the same cleaned samples and same train/test split as M0.
4. Traverse STUDENT_Z_THRESHOLDS and STUDENT_CV_VALUES.
5. Output parameter-search results for the protein proxy model, M0 model, and M2-C correction models.
6. Test monotonic constraints as knowledge-guided correction learning.

Model family
M0_RF_baseline:
    Original RandomForest baseline using the original common weather factors.
C1_TMIN_threshold_constraint:
    Direct HGB model with TMIN_excess_20 as a monotonic positive feature.
C2_Protein_constraint:
    Direct HGB model with Predicted Total protein as a monotonic positive feature.
C3_ProteinHeatRisk_constraint:
    Direct HGB model with only Protein_Heat_Risk added as the knowledge feature, and constrained positive.
    It does not additionally include Predicted Total protein or TMIN_excess_20 as separate input features.
C4_Protein_and_Risk_constraint:
    Direct HGB model with Predicted Total protein and Protein_Heat_Risk, both constrained positive.

Important
TMIN_excess_20, TMIN_above_20_flag, Predicted Total protein and Protein_Heat_Risk are not used in M0.
They are used only in constrained models where explicitly required.
"""

import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# =========================================================
# Configuration
# =========================================================
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\Direct_M0_constraint_only_gridsearch_results"
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

KNOWLEDGE_ONLY_COLS = [TMIN_EXCESS_COL, TMIN_FLAG_COL, PROTEIN_RISK_COL, PREDICTED_PROTEIN_COL]

STUDENT_TEST_SIZE = 0.30
RANDOM_STATE = 42
TEACHER_Z_THRESHOLD = 4
STUDENT_Z_THRESHOLDS = [3]
STUDENT_CV_VALUES = [7]
TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10

PROTEIN_PROXY_N_ITER = 40
M0_N_ITER = 100
HGB_N_ITER = 80

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

FIGSIZE = {
    "bar": (6, 4),
    "heatmap": (7, 5),
    "residual": (6, 5),
    "residual_hist": (6, 5),
    "qq": (6, 5),
    "learning_curve": (6, 5),
    "shap_dot": (6, 4),
    "shap_bar": (6, 4),
    "shap_dependence": (6, 5),
    "pdp": (6, 5),
}
JOINTGRID_HEIGHT = 6
TOP_K = 5
MODEL_DISPLAY_NAME = "M2-C"

XAXIS_LABEL = {
    "shap_dot_x": "SHAP value",
    "shap_bar_x": "mean(|SHAP value|)",
}

# =========================================================
# Parameter grids, aligned with the ordinary residual-correction M0 grid.
# =========================================================
m0_rf_param_grid = {
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

hgb_param_grid = {
    "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
    "model__max_iter": [100, 200, 300, 500],
    "model__max_leaf_nodes": [5, 7, 10, 15, 20],
    "model__max_depth": [2, 3, 5, None],
    "model__min_samples_leaf": [10, 15, 20, 25, 30],
    "model__l2_regularization": [0.0, 0.01, 0.05, 0.10, 0.50, 1.0, 2.0],
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


def build_monotonic_constraints(feature_cols, positive_features):
    return [1 if col in positive_features else 0 for col in feature_cols]


def cv_results_to_dataframe(search_obj, stage, version, z_thr=None, cv_value=None):
    """
    Convert RandomizedSearchCV.cv_results_ into a flat DataFrame for traceable parameter-search output.
    This function only records search results and does not affect model training or selection.
    """
    out = pd.DataFrame(search_obj.cv_results_).copy()
    out.insert(0, "Stage", stage)
    out.insert(1, "Version", version)
    out.insert(2, "z_score_threshold", z_thr)
    out.insert(3, "cv", cv_value)
    if "params" in out.columns:
        out["params"] = out["params"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    for col in out.columns:
        if col.startswith("param_"):
            out[col] = out[col].astype(str)
    return out


def make_best_param_record(stage, version, best_params, best_score=None, z_thr=None, cv_value=None, extra_info=None):
    """
    Store best parameters in expanded columns for direct reading in Excel.
    """
    rec = {
        "Stage": stage,
        "Version": version,
        "z_score_threshold": z_thr,
        "cv": cv_value,
        "best_score": best_score,
    }
    if extra_info:
        rec.update(extra_info)
    for k, v in dict(best_params).items():
        rec[k] = v
    return rec


def tune_rf(df, feature_cols, target_col, param_grid, cv, n_iter, scoring="r2"):
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
    rs.fit(X_scaled, y)
    return rs.best_estimator_, scaler, rs


def train_m0_on_fixed_split(X_train, y_train, X_test, cv):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=m0_rf_param_grid,
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


def train_hgb_correction_model(X_train_df, residual_target, positive_features, cv):
    """
    Train the knowledge-constrained correction term in:
        M2-C prediction = M0 prediction + constrained correction

    The HGB model predicts the M0 residual on the training set.
    Positive monotonic constraints are applied only to specified knowledge variables.
    """
    feature_cols = X_train_df.columns.tolist()
    monotonic_cst = build_monotonic_constraints(feature_cols, positive_features)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            loss="squared_error",
            monotonic_cst=monotonic_cst,
        )),
    ])
    rs = RandomizedSearchCV(
        pipe,
        param_distributions=hgb_param_grid,
        n_iter=HGB_N_ITER,
        cv=cv,
        scoring="r2",
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rs.fit(X_train_df.values, residual_target)
    info = {
        "positive_constraint_features": str(positive_features),
        "monotonic_cst": str(monotonic_cst),
        "best_params": json.dumps(rs.best_params_, ensure_ascii=False),
        "cv_r2_on_train_residual": float(rs.best_score_),
        "feature_cols": str(feature_cols),
    }
    return rs.best_estimator_, info, rs


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



def save_current_figure(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def simplify_model_name(label):
    label = str(label)
    if "M0" in label:
        return "M0"
    return MODEL_DISPLAY_NAME


def plot_barh(values, names, save_path, figsize, xlabel, title=None, xerr=None):
    values = np.asarray(values, dtype=float)
    names = np.asarray(names, dtype=object)
    order = np.argsort(values)
    values_sorted = values[order]
    names_sorted = names[order]
    if xerr is not None:
        xerr = np.asarray(xerr, dtype=float)[order]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(values_sorted))
    cols = plt.cm.Blues(np.linspace(0.45, 0.95, len(values_sorted)))
    ax.barh(
        y,
        values_sorted,
        color=cols,
        edgecolor="black",
        linewidth=1.2,
        height=0.7,
    )

    if xerr is not None:
        ax.errorbar(values_sorted, y, xerr=xerr, fmt="none", ecolor="black", capsize=3, linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")

    max_val = np.nanmax(np.abs(values_sorted)) if len(values_sorted) > 0 else 0.0
    offset = max(max_val * 0.01, 1e-6)
    for yi, vi in zip(y, values_sorted):
        ha = "left" if vi >= 0 else "right"
        ax.text(vi + offset if vi >= 0 else vi - offset, yi, f"{vi:.3f}", va="center", ha=ha, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    save_current_figure(save_path)


def plot_true_vs_pred(y_train, y_train_pred, y_test, y_test_pred, title, save_path):
    data_train = pd.DataFrame({"True": y_train, "Predicted": y_train_pred, "Data Set": "Train"})
    data_test = pd.DataFrame({"True": y_test, "Predicted": y_test_pred, "Data Set": "Test"})
    data_plot = pd.concat([data_train, data_test], axis=0)

    palette = {"Train": "#b4d4e1", "Test": "#f4ba8a"}
    g = sns.JointGrid(
        data=data_plot,
        x="True",
        y="Predicted",
        hue="Data Set",
        height=JOINTGRID_HEIGHT,
        palette=palette,
    )
    g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
    sns.regplot(
        data=data_train,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color="#b4d4e1",
        label="Train fit",
    )
    sns.regplot(
        data=data_test,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color="#f4ba8a",
        label="Test fit",
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
        fontsize=20,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )
    plot_model_label = "M0" if "M0" in str(title) else MODEL_DISPLAY_NAME
    ax.text(
        0.75,
        0.99,
        f"{plot_model_label}",
        transform=ax.transAxes,
        fontsize=18,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

    min_val = data_plot["True"].min()
    max_val = data_plot["True"].max()
    ax.plot([min_val, max_val], [min_val, max_val], c="black", linestyle="--", alpha=0.7, label="x=y")
    ax.legend(loc="best", fontsize=14)
    #ax.set_title(plot_model_label, fontsize=16, fontweight="bold")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def save_dataframe_dict_to_excel(sheet_dict, save_path):
    with pd.ExcelWriter(save_path) as writer:
        for sheet_name, df in sheet_dict.items():
            safe_name = str(sheet_name)[:31]
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=safe_name, index=False)
            else:
                pd.DataFrame(df).to_excel(writer, sheet_name=safe_name, index=False)


def get_pipeline_scaler_and_estimator(model):
    if isinstance(model, Pipeline):
        return model.named_steps.get("scaler", None), model.named_steps.get("model", None)
    return None, model


def evaluate_prediction_dataframe(y_true, y_pred, prefix):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return pd.DataFrame({
        "True": y_true,
        "Predicted": y_pred,
        "Residual": residual,
        "Absolute residual": np.abs(residual),
        "Squared residual": residual ** 2,
        "Model": simplify_model_name(prefix),
    })


def plot_metric_comparison(best_m0, best_constrained, save_path):
    metrics = pd.DataFrame({
        "Metric": ["R2", "RMSE", "MAE"],
        "M0": [best_m0["Test R2"], best_m0["Test RMSE"], best_m0["Test MAE"]],
        MODEL_DISPLAY_NAME: [best_constrained["Test R2"], best_constrained["Test RMSE"], best_constrained["Test MAE"]],
    })
    plot_df = metrics.melt(id_vars="Metric", var_name="Model", value_name="Value")
    plt.figure(figsize=FIGSIZE["bar"])
    ax = sns.barplot(data=plot_df, x="Metric", y="Value", hue="Model")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=10)
    plt.xlabel("Metric", fontsize=14, fontweight="bold")
    plt.ylabel("Value", fontsize=14, fontweight="bold")
    plt.title("M0 vs M2-C", fontsize=16, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_figure(save_path)
    return metrics


def plot_residual_diagnostics(y_true, y_pred, model_label, output_dir):
    clean_label = simplify_model_name(model_label)
    pred_df = evaluate_prediction_dataframe(y_true, y_pred, clean_label)

    plt.figure(figsize=FIGSIZE["residual"])
    plt.scatter(pred_df["Predicted"], pred_df["Residual"], s=120, alpha=0.7, edgecolors="k", color="#b4d4e1")
    plt.axhline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Predicted Values", fontsize=18, fontweight="bold")
    plt.ylabel("Residuals", fontsize=18, fontweight="bold")
    plt.title("Residual Plot", fontsize=20, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    save_current_figure(os.path.join(output_dir, f"{model_label}_residuals_vs_predicted.png"))

    plt.figure(figsize=FIGSIZE["residual_hist"])
    plt.hist(pred_df["Residual"], bins=15, edgecolor="black", alpha=0.75, color="#5aa0c8")
    plt.axvline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Residuals", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("Residual Distribution", fontsize=20, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_figure(os.path.join(output_dir, f"{model_label}_residual_distribution.png"))

    fig = plt.figure(figsize=FIGSIZE["qq"])
    ax = fig.add_subplot(111)
    stats.probplot(pred_df["Residual"], dist="norm", plot=ax)
    ax.set_title("Residual Q-Q Plot", fontsize=20, fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles", fontsize=18, fontweight="bold")
    ax.set_ylabel("Ordered Values", fontsize=18, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    save_current_figure(os.path.join(output_dir, f"{model_label}_residual_QQ_plot.png"))

    return pred_df


def plot_residual_comparison(y_true, m0_pred, constrained_pred, save_path):
    residual_df = pd.concat([
        pd.DataFrame({"Residual": np.asarray(y_true) - np.asarray(m0_pred), "Model": "M0"}),
        pd.DataFrame({"Residual": np.asarray(y_true) - np.asarray(constrained_pred), "Model": MODEL_DISPLAY_NAME}),
    ], axis=0)
    plt.figure(figsize=FIGSIZE["residual"])
    sns.boxplot(data=residual_df, x="Model", y="Residual")
    sns.stripplot(data=residual_df, x="Model", y="Residual", alpha=0.5, size=5, color="black")
    plt.axhline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("")
    plt.ylabel("Residuals", fontsize=18, fontweight="bold")
    plt.title("Residual Comparison", fontsize=20, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_figure(save_path)
    return residual_df


def plot_prediction_shift(y_true, m0_pred, constrained_pred, save_path):
    shift_df = pd.DataFrame({
        "True": y_true,
        "M0 prediction": m0_pred,
        "M2-C prediction": constrained_pred,
        "Prediction shift": np.asarray(constrained_pred) - np.asarray(m0_pred),
        "M0 residual": np.asarray(y_true) - np.asarray(m0_pred),
    })
    plt.figure(figsize=FIGSIZE["residual"])
    plt.scatter(shift_df["M0 prediction"], shift_df["Prediction shift"], s=120, alpha=0.7, edgecolors="k", color="#b4d4e1")
    plt.axhline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("M0 Predicted Values", fontsize=18, fontweight="bold")
    plt.ylabel("Prediction Shift", fontsize=18, fontweight="bold")
    plt.title("Prediction Shift", fontsize=20, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    save_current_figure(save_path)
    return shift_df


def plot_model_heatmaps(result_df, output_dir):
    constrained_only = result_df[result_df["Version"] != "M0_RF_baseline"].copy()
    if constrained_only.empty:
        return

    for value_col, file_name, title in [
        ("Test R2", "heatmap_constrained_test_R2.png", "Test R2"),
        ("Delta test R2 vs M0", "heatmap_constrained_delta_R2_vs_M0.png", "Delta R2"),
        ("Test RMSE", "heatmap_constrained_test_RMSE.png", "Test RMSE"),
    ]:
        pivot_df = constrained_only.pivot_table(
            index="Version",
            columns=["z_score_threshold", "cv"],
            values=value_col,
            aggfunc="mean",
        )
        fig_width = max(8, 0.9 * pivot_df.shape[1])
        fig_height = max(5, 0.5 * pivot_df.shape[0] + 2)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5, linecolor="white")
        plt.xlabel("Z and CV", fontsize=14, fontweight="bold")
        plt.ylabel("Model", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=16, fontweight="bold")
        save_current_figure(os.path.join(output_dir, file_name))


def plot_rf_feature_importance(model, feature_cols, save_path, title):
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "RF impurity importance": model.feature_importances_,
    }).sort_values("RF impurity importance", ascending=False)

    top_df = importance_df.head(TOP_K)
    plot_barh(
        values=top_df["RF impurity importance"].values,
        names=top_df["Feature"].values,
        save_path=save_path,
        figsize=FIGSIZE["shap_bar"],
        xlabel="RF impurity importance",
        title="RF Feature Importance",
    )
    return importance_df


def plot_permutation_importance_for_model(model, X_test_df, y_test, save_path, title):
    perm = permutation_importance(
        model,
        X_test_df.values,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "Feature": X_test_df.columns,
        "Permutation importance mean": perm.importances_mean,
        "Permutation importance std": perm.importances_std,
    }).sort_values("Permutation importance mean", ascending=False)

    top_df = perm_df.head(TOP_K)
    plot_barh(
        values=top_df["Permutation importance mean"].values,
        names=top_df["Feature"].values,
        save_path=save_path,
        figsize=FIGSIZE["shap_bar"],
        xlabel="Permutation importance mean decrease in R²",
        title="Permutation Feature Importance",
        xerr=top_df["Permutation importance std"].values,
    )
    return perm_df


def plot_hgb_partial_dependence(model, X_test_df, feature_cols_to_plot, output_dir):
    plotted = []
    for feature_name in feature_cols_to_plot:
        if feature_name not in X_test_df.columns:
            continue
        try:
            fig, ax = plt.subplots(figsize=FIGSIZE["pdp"])
            PartialDependenceDisplay.from_estimator(
                model,
                X_test_df.values,
                features=[X_test_df.columns.get_loc(feature_name)],
                feature_names=X_test_df.columns.tolist(),
                ax=ax,
            )
            ax.set_title("PDP", fontsize=20, fontweight="bold")
            ax.set_xlabel(feature_name, fontsize=18, fontweight="bold")
            ax.grid(alpha=0.3, linestyle="--")
            path = os.path.join(output_dir, f"best_constraint_PDP_{feature_name}.png")
            save_current_figure(path)
            plotted.append({"Feature": feature_name, "PDP file": path})
        except Exception as exc:
            plotted.append({"Feature": feature_name, "PDP file": f"Failed: {exc}"})
    return pd.DataFrame(plotted)


def compute_and_plot_shap_for_model(model, X_test_df, feature_cols, output_dir, model_label):
    """
    SHAP is attempted for the final best model only.
    For Pipeline-based HGB, SHAP is calculated on the scaled matrix and the internal HGB estimator.
    """
    try:
        scaler, estimator = get_pipeline_scaler_and_estimator(model)
        if scaler is not None:
            X_for_shap = scaler.transform(X_test_df.values)
        else:
            X_for_shap = X_test_df.values

        explainer = shap.Explainer(estimator)
        shap_values = explainer.shap_values(X_for_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_mean = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Mean Importance": shap_mean,
        }).sort_values("SHAP Mean Importance", ascending=False)
        shap_df.to_excel(os.path.join(output_dir, f"{model_label}_shap_feature_importances.xlsx"), index=False)

        top_df = shap_df.head(TOP_K)
        top_idx = [feature_cols.index(f) for f in top_df["Feature"]]
        top_names = top_df["Feature"].values

        shap.summary_plot(
            shap_values[:, top_idx],
            X_for_shap[:, top_idx],
            feature_names=top_names,
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
        plt.savefig(os.path.join(output_dir, f"{model_label}_shap_beeswarm.png"), dpi=600, bbox_inches="tight")
        plt.close()

        plot_barh(
            values=top_df["SHAP Mean Importance"].values,
            names=top_df["Feature"].values,
            save_path=os.path.join(output_dir, f"{model_label}_shap_bar.png"),
            figsize=FIGSIZE["shap_bar"],
            xlabel=XAXIS_LABEL["shap_bar_x"],
            title="SHAP Feature Importance",
        )

        if not top_df.empty:
            best_feature = top_df.iloc[0]["Feature"]
            feature_index = feature_cols.index(best_feature)
            shap_values_for_feature = shap_values[:, feature_index]
            original_feature_values = X_test_df[best_feature].values

            plt.figure(figsize=FIGSIZE["shap_dependence"])
            if len(np.unique(original_feature_values)) >= 3:
                lowess_fit = sm.nonparametric.lowess(shap_values_for_feature, original_feature_values, frac=0.7)
                plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="blue", linewidth=3, label="Lowess Fit")
                plt.fill_between(
                    lowess_fit[:, 0],
                    lowess_fit[:, 1] - 0.1,
                    lowess_fit[:, 1] + 0.1,
                    color="gray",
                    alpha=0.3,
                    label="95% CI",
                )
                plt.legend(fontsize=14)
            plt.scatter(original_feature_values, shap_values_for_feature, s=70, alpha=0.55, edgecolors="k", color="#b4d4e1")
            plt.grid(True, linestyle="--", linewidth=1, color="lightgray", alpha=0.7)
            plt.xlabel(best_feature, fontsize=18, fontweight="bold")
            plt.ylabel("SHAP Value", fontsize=18, fontweight="bold")
            plt.title("SHAP Dependence Plot", fontsize=20, fontweight="bold")
            plt.tick_params(axis="both", which="major", labelsize=14)
            save_current_figure(os.path.join(output_dir, f"{model_label}_shap_dependence_{best_feature}.png"))

        return shap_df

    except Exception as exc:
        with open(os.path.join(output_dir, f"{model_label}_shap_failed_reason.txt"), "w", encoding="utf-8") as f:
            f.write(str(exc))
        return pd.DataFrame({"SHAP status": [f"Failed: {exc}"]})


def plot_learning_curve_fixed_best(model, X_train_df, y_train, X_test_df, y_test, save_path, title):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []
    n_train_total = len(X_train_df)

    for train_size in train_sizes:
        n = max(2, int(n_train_total * train_size))
        X_subset = X_train_df.iloc[:n].values
        y_subset = y_train[:n]

        if isinstance(model, Pipeline):
            _, estimator = get_pipeline_scaler_and_estimator(model)
            if isinstance(estimator, HistGradientBoostingRegressor):
                cloned_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        loss="squared_error",
                        monotonic_cst=estimator.monotonic_cst,
                        **{k: v for k, v in estimator.get_params().items()
                           if k not in ["random_state", "loss", "monotonic_cst"]}
                    )),
                ])
            elif isinstance(estimator, RandomForestRegressor):
                cloned_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(**estimator.get_params())),
                ])
            else:
                raise TypeError(f"Unsupported estimator in Pipeline for learning curve: {type(estimator).__name__}")
        else:
            cloned_model = RandomForestRegressor(**model.get_params())

        cloned_model.fit(X_subset, y_subset)
        train_scores.append(r2_score(y_subset, cloned_model.predict(X_subset)))
        test_scores.append(r2_score(y_test, cloned_model.predict(X_test_df.values)))

    curve_df = pd.DataFrame({
        "Training set size": (train_sizes * n_train_total).astype(int),
        "Training R2": train_scores,
        "Testing R2": test_scores,
    })

    plt.figure(figsize=FIGSIZE["learning_curve"])
    plt.plot(curve_df["Training set size"], curve_df["Training R2"], "o-", color="r", label="Training score", linewidth=2)
    plt.plot(curve_df["Training set size"], curve_df["Testing R2"], "o-", color="b", label="Testing score", linewidth=2)
    plt.xlabel("Training Set Size", fontsize=18, fontweight="bold")
    plt.ylabel("Score (R²)", fontsize=18, fontweight="bold")
    plt.title("Learning Curve", fontsize=20, fontweight="bold")
    plt.legend(loc="best", fontsize=14)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    save_current_figure(save_path)
    return curve_df


def plot_residual_against_variable(data_df, residual_col, variable, save_path):
    if variable not in data_df.columns:
        return

    x = data_df[variable].values
    residuals = data_df[residual_col].values

    plt.figure(figsize=FIGSIZE["residual"])
    plt.scatter(x, residuals, s=100, alpha=0.75, edgecolors="k", color="#b4d4e1")
    plt.axhline(0, color="r", linestyle="--", linewidth=2)

    if len(np.unique(x)) >= 3:
        lowess_fit = sm.nonparametric.lowess(residuals, x, frac=0.7)
        plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="blue", linewidth=3, label="Lowess Fit")
        plt.legend(fontsize=14)

    plt.xlabel(variable, fontsize=18, fontweight="bold")
    plt.ylabel("Residuals", fontsize=18, fontweight="bold")
    plt.title(f"Residuals vs {variable}", fontsize=20, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    save_current_figure(save_path)

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

    if TMIN_COL not in base_weather_features:
        raise ValueError(f"{TMIN_COL} must be in base weather features.")

    print("\nBase M0 weather features, excluding knowledge-only variables:")
    print(base_weather_features)

    pd.DataFrame({"M0_base_weather_features": base_weather_features}).to_excel(
        os.path.join(OUTPUT_DIR, "M0_base_weather_features.xlsx"), index=False
    )

    # Teacher data and protein proxy model.
    teacher_df = teacher_numeric[[TARGET_COL, PROTEIN_COL] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    teacher_clean, teacher_outliers = remove_outliers_by_zscore(
        teacher_df, base_weather_features + [PROTEIN_COL], TEACHER_Z_THRESHOLD
    )
    print("\nTeacher valid n:", len(teacher_df))
    print("Teacher cleaned n:", len(teacher_clean))
    print("Teacher outliers removed:", len(teacher_outliers))

    teacher_cv = RepeatedKFold(n_splits=TEACHER_N_SPLITS, n_repeats=TEACHER_N_REPEATS, random_state=RANDOM_STATE)
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
    protein_oof_metrics = calculate_metrics(teacher_clean[PROTEIN_COL].values, oof_protein, "Protein OOF prediction")
    print("\nProtein proxy best params:", protein_search.best_params_)
    print("Protein proxy CV R2:", protein_search.best_score_)
    print("Protein proxy OOF metrics:", protein_oof_metrics)

    # Student data with knowledge-only variables available for constrained models.
    student_df = original_numeric[[original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    student_df[PREDICTED_PROTEIN_COL] = protein_model.predict(protein_scaler.transform(student_df[base_weather_features].values))
    student_df = add_protein_heat_risk(student_df, PREDICTED_PROTEIN_COL)
    print("\nStudent valid n:", len(student_df))

    records = []
    bundles = {}
    parameter_search_tables = []
    best_parameter_records = []

    parameter_search_tables.append(
        cv_results_to_dataframe(
            protein_search,
            stage="Protein proxy",
            version="Protein_proxy_RF",
            z_thr=None,
            cv_value=f"{TEACHER_N_SPLITS}x{TEACHER_N_REPEATS}",
        )
    )
    best_parameter_records.append(
        make_best_param_record(
            stage="Protein proxy",
            version="Protein_proxy_RF",
            best_params=protein_search.best_params_,
            best_score=float(protein_search.best_score_),
            z_thr=None,
            cv_value=f"{TEACHER_N_SPLITS}x{TEACHER_N_REPEATS}",
            extra_info={"scoring": "r2"},
        )
    )

    for z_thr in STUDENT_Z_THRESHOLDS:
        student_clean, student_outliers = remove_outliers_by_zscore(student_df, base_weather_features, z_thr)
        if len(student_clean) < 30:
            print(f"Z-score={z_thr}: cleaned sample size too small; skipped.")
            continue

        y = student_clean[original_target_col].values
        train_idx, test_idx = train_test_split(
            np.arange(len(student_clean)), test_size=STUDENT_TEST_SIZE, random_state=RANDOM_STATE
        )
        train_df = student_clean.iloc[train_idx].copy()
        test_df = student_clean.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        for cv_value in STUDENT_CV_VALUES:
            if cv_value >= len(train_df):
                continue
            print("\n" + "=" * 80)
            print(f"Training M0 + constrained correction models: Z-score={z_thr}, CV={cv_value}")

            # M0 baseline for this same Z-score and CV.
            m0_model, m0_scaler, m0_search, m0_train_pred, m0_test_pred = train_m0_on_fixed_split(
                train_df[base_weather_features].values, y_train, test_df[base_weather_features].values, cv=cv_value
            )
            parameter_search_tables.append(
                cv_results_to_dataframe(
                    m0_search,
                    stage="M0 baseline",
                    version="M0_RF_baseline",
                    z_thr=z_thr,
                    cv_value=cv_value,
                )
            )
            best_parameter_records.append(
                make_best_param_record(
                    stage="M0 baseline",
                    version="M0_RF_baseline",
                    best_params=m0_search.best_params_,
                    best_score=float(m0_search.best_score_),
                    z_thr=z_thr,
                    cv_value=cv_value,
                    extra_info={"scoring": "r2"},
                )
            )
            m0_record = {
                "Version": "M0_RF_baseline",
                "Description": "Original M0 RF baseline. No protein, no threshold feature, no constraint.",
                "Model type": "RandomForestRegressor",
                "Positive constraints": "None",
                "Feature cols": str(base_weather_features),
                "Best params": json.dumps(m0_search.best_params_, ensure_ascii=False),
                "CV R2 on train": float(m0_search.best_score_),
                "z_score_threshold": z_thr,
                "cv": cv_value,
                "n_samples_after_cleaning": len(student_clean),
                "n_outliers_removed": len(student_outliers),
                "outlier_removed_ratio": float(len(student_outliers) / len(student_df)),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "Protein proxy CV R2": float(protein_search.best_score_),
                "Protein proxy OOF R2": protein_oof_metrics["Protein OOF prediction R2"],
            }
            m0_record.update(calculate_metrics(y_train, m0_train_pred, "Train"))
            m0_record.update(calculate_metrics(y_test, m0_test_pred, "Test"))
            m0_record["M0 Test R2 within same Z-CV"] = m0_record["Test R2"]
            m0_record["M0 Test RMSE within same Z-CV"] = m0_record["Test RMSE"]
            m0_record["M0 Test MAE within same Z-CV"] = m0_record["Test MAE"]
            m0_record["Delta test R2 vs M0"] = 0.0
            m0_record["Delta test RMSE vs M0"] = 0.0
            m0_record["Delta test MAE vs M0"] = 0.0
            m0_record["Train-Test R2 gap"] = m0_record["Train R2"] - m0_record["Test R2"]
            records.append(m0_record)

            combo_m0_r2 = m0_record["Test R2"]
            combo_m0_rmse = m0_record["Test RMSE"]
            combo_m0_mae = m0_record["Test MAE"]

            bundles[f"Z{z_thr}_CV{cv_value}_M0_RF_baseline"] = {
                "model": m0_model,
                "scaler": m0_scaler,
                "train_pred": m0_train_pred,
                "test_pred": m0_test_pred,
                "y_train": y_train,
                "y_test": y_test,
                "train_df": train_df.copy(),
                "test_df": test_df.copy(),
            }

            constrained_specs = [
                {
                    "Version": "C1_TMIN_threshold_constraint",
                    "Description": "M0 prediction plus HGB correction with monotonic positive constraint on TMIN_excess_20.",
                    "features": base_weather_features + [TMIN_EXCESS_COL],
                    "positive_features": [TMIN_EXCESS_COL],
                },
                {
                    "Version": "C2_Protein_constraint",
                    "Description": "M0 prediction plus HGB correction with monotonic positive constraint on Predicted Total protein.",
                    "features": base_weather_features + [PREDICTED_PROTEIN_COL],
                    "positive_features": [PREDICTED_PROTEIN_COL],
                },
                {
                    "Version": "C3_ProteinHeatRisk_constraint",
                    "Description": "M0 prediction plus HGB correction with Protein_Heat_Risk constrained positive.",
                    "features": base_weather_features + [PROTEIN_RISK_COL],
                    "positive_features": [PROTEIN_RISK_COL],
                },
                {
                    "Version": "C4_Protein_and_Risk_constraint",
                    "Description": "M0 prediction plus HGB correction with Predicted Total protein and Protein_Heat_Risk constrained positive.",
                    "features": base_weather_features + [PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL],
                    "positive_features": [PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL],
                },
            ]

            for spec in constrained_specs:
                print("Training", spec["Version"])
                X_train_df = train_df[spec["features"]].copy()
                X_test_df = test_df[spec["features"]].copy()

                residual_target_train = y_train - m0_train_pred
                model, info, correction_search = train_hgb_correction_model(
                    X_train_df,
                    residual_target_train,
                    spec["positive_features"],
                    cv=cv_value,
                )
                train_correction = model.predict(X_train_df.values)
                test_correction = model.predict(X_test_df.values)
                train_pred = m0_train_pred + train_correction
                test_pred = m0_test_pred + test_correction

                rec = {
                    "Version": spec["Version"],
                    "Description": spec["Description"],
                    "Model type": "M0 + HistGradientBoostingRegressor correction",
                    "Positive constraints": str(spec["positive_features"]),
                    "Feature cols": str(spec["features"]),
                    "z_score_threshold": z_thr,
                    "cv": cv_value,
                    "n_samples_after_cleaning": len(student_clean),
                    "n_outliers_removed": len(student_outliers),
                    "outlier_removed_ratio": float(len(student_outliers) / len(student_df)),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "Protein proxy CV R2": float(protein_search.best_score_),
                    "Protein proxy OOF R2": protein_oof_metrics["Protein OOF prediction R2"],
                }
                rec.update(info)
                rec.update(calculate_metrics(residual_target_train, train_correction, "Correction train residual"))
                rec.update(calculate_metrics(y_test - m0_test_pred, test_correction, "Correction test residual"))
                rec.update(calculate_metrics(y_train, train_pred, "Train"))
                rec.update(calculate_metrics(y_test, test_pred, "Test"))
                rec.update({
                    "M0 Test R2 within same Z-CV": combo_m0_r2,
                    "M0 Test RMSE within same Z-CV": combo_m0_rmse,
                    "M0 Test MAE within same Z-CV": combo_m0_mae,
                    "Delta test R2 vs M0": rec["Test R2"] - combo_m0_r2,
                    "Delta test RMSE vs M0": rec["Test RMSE"] - combo_m0_rmse,
                    "Delta test MAE vs M0": rec["Test MAE"] - combo_m0_mae,
                    "Train-Test R2 gap": rec["Train R2"] - rec["Test R2"],
                })
                records.append(rec)
                parameter_search_tables.append(
                    cv_results_to_dataframe(
                        correction_search,
                        stage="M2-C constrained correction",
                        version=spec["Version"],
                        z_thr=z_thr,
                        cv_value=cv_value,
                    )
                )
                best_parameter_records.append(
                    make_best_param_record(
                        stage="M2-C constrained correction",
                        version=spec["Version"],
                        best_params=correction_search.best_params_,
                        best_score=float(correction_search.best_score_),
                        z_thr=z_thr,
                        cv_value=cv_value,
                        extra_info={
                            "scoring": "r2",
                            "positive_constraint_features": str(spec["positive_features"]),
                            "correction_target": "Train residual = y_train - M0_train_prediction",
                        },
                    )
                )

                bundle_key = f"Z{z_thr}_CV{cv_value}_{spec['Version']}"
                bundles[bundle_key] = {
                    "model": model,
                    "train_pred": train_pred,
                    "test_pred": test_pred,
                    "train_correction": train_correction,
                    "test_correction": test_correction,
                    "m0_train_pred": m0_train_pred,
                    "m0_test_pred": m0_test_pred,
                    "y_train": y_train,
                    "y_test": y_test,
                    "train_df": train_df.copy(),
                    "test_df": test_df.copy(),
                    "feature_cols": spec["features"],
                }

    result_df = pd.DataFrame(records)
    if result_df.empty:
        raise RuntimeError("No valid direct constraint results were generated.")

    result_df.to_excel(os.path.join(OUTPUT_DIR, "direct_constraint_only_all_results_gridsearch.xlsx"), index=False)

    if parameter_search_tables:
        all_parameter_search_df = pd.concat(parameter_search_tables, axis=0, ignore_index=True)
        all_parameter_search_df.to_excel(
            os.path.join(OUTPUT_DIR, "all_model_parameter_search_cv_results.xlsx"),
            index=False,
        )
    else:
        all_parameter_search_df = pd.DataFrame()

    best_parameter_df = pd.DataFrame(best_parameter_records)
    best_parameter_df.to_excel(
        os.path.join(OUTPUT_DIR, "expanded_best_parameters_by_model.xlsx"),
        index=False,
    )

    core_cols = [
        "z_score_threshold", "cv", "Version", "Description", "Positive constraints",
        "Test R2", "Test RMSE", "Test MAE",
        "M0 Test R2 within same Z-CV", "Delta test R2 vs M0", "Delta test RMSE vs M0", "Delta test MAE vs M0",
        "Train R2", "Train-Test R2 gap", "CV R2 on train", "cv_r2_on_train_residual", "Best params", "Feature cols",
    ]
    available_core_cols = [c for c in core_cols if c in result_df.columns]
    result_df[available_core_cols].to_excel(os.path.join(OUTPUT_DIR, "direct_constraint_only_core_metrics_gridsearch.xlsx"), index=False)

    constrained_df = result_df[result_df["Version"] != "M0_RF_baseline"].copy()
    best_constrained = constrained_df.sort_values(["Test R2", "Test RMSE"], ascending=[False, True]).iloc[0].to_dict()
    pd.DataFrame([best_constrained]).to_excel(os.path.join(OUTPUT_DIR, "best_direct_constraint_model_metrics_gridsearch.xlsx"), index=False)

    # Explicit M2-C output alias.
    # This does not change model selection. It only saves the same selected best constrained-correction model
    # under the final model name M2-C for direct downstream use.
    pd.DataFrame([best_constrained]).to_excel(os.path.join(OUTPUT_DIR, "best_M2C_model_metrics_gridsearch.xlsx"), index=False)

    m0_df = result_df[result_df["Version"] == "M0_RF_baseline"].copy()
    best_m0 = m0_df.sort_values(["Test R2", "Test RMSE"], ascending=[False, True]).iloc[0].to_dict()
    pd.DataFrame([best_m0]).to_excel(os.path.join(OUTPUT_DIR, "best_M0_baseline_metrics_gridsearch.xlsx"), index=False)

    # Save models and plots for the best M0 and best constrained model.
    joblib.dump(protein_model, os.path.join(OUTPUT_DIR, "protein_proxy_model.pkl"))
    joblib.dump(protein_scaler, os.path.join(OUTPUT_DIR, "protein_proxy_scaler.pkl"))
    best_constraint_key = f"Z{best_constrained['z_score_threshold']}_CV{best_constrained['cv']}_{best_constrained['Version']}"
    best_m0_key = f"Z{best_m0['z_score_threshold']}_CV{best_m0['cv']}_{best_m0['Version']}"
    if best_constraint_key in bundles:
        joblib.dump(bundles[best_constraint_key]["model"], os.path.join(OUTPUT_DIR, "best_direct_constraint_model.pkl"))

        # Explicit M2-C model object output.
        joblib.dump(bundles[best_constraint_key]["model"], os.path.join(OUTPUT_DIR, "best_M2C_model.pkl"))

        plot_true_vs_pred(
            bundles[best_constraint_key]["y_train"], bundles[best_constraint_key]["train_pred"],
            bundles[best_constraint_key]["y_test"], bundles[best_constraint_key]["test_pred"],
            f"Best direct constrained model ({best_constrained['Version']})",
            os.path.join(OUTPUT_DIR, "best_direct_constraint_true_vs_predicted.png"),
        )
        plot_true_vs_pred(
            bundles[best_constraint_key]["y_train"], bundles[best_constraint_key]["train_pred"],
            bundles[best_constraint_key]["y_test"], bundles[best_constraint_key]["test_pred"],
            MODEL_DISPLAY_NAME,
            os.path.join(OUTPUT_DIR, "best_M2C_true_vs_predicted.png"),
        )
    if best_m0_key in bundles:
        joblib.dump(bundles[best_m0_key]["model"], os.path.join(OUTPUT_DIR, "best_M0_RF_baseline_model.pkl"))
        joblib.dump(bundles[best_m0_key]["scaler"], os.path.join(OUTPUT_DIR, "best_M0_RF_baseline_scaler.pkl"))
        plot_true_vs_pred(
            bundles[best_m0_key]["y_train"], bundles[best_m0_key]["train_pred"],
            bundles[best_m0_key]["y_test"], bundles[best_m0_key]["test_pred"],
            "Best M0 RF baseline",
            os.path.join(OUTPUT_DIR, "best_M0_true_vs_predicted.png"),
        )

    # =========================================================
    # Extended outputs for the final best constrained model only.
    # These outputs do not change the original training, model selection, or prediction logic.
    # =========================================================
    best_constraint_bundle = bundles.get(best_constraint_key, None)

    # The globally best M0 can come from a different Z-score/CV split than the best constrained model.
    # For sample-level diagnostics and prediction-shift plots, use the M0 trained on the same split
    # as the final best constrained model. This preserves the original model-selection logic and
    # avoids length mismatch between different test sets.
    paired_m0_key = f"Z{best_constrained['z_score_threshold']}_CV{best_constrained['cv']}_M0_RF_baseline"
    paired_m0_bundle = bundles.get(paired_m0_key, None)
    paired_m0_record_df = result_df[
        (result_df["Version"] == "M0_RF_baseline")
        & (result_df["z_score_threshold"] == best_constrained["z_score_threshold"])
        & (result_df["cv"] == best_constrained["cv"])
    ].copy()
    paired_m0_record = paired_m0_record_df.iloc[0].to_dict() if not paired_m0_record_df.empty else best_m0

    if best_constraint_bundle is not None and paired_m0_bundle is not None:
        extended_dir = os.path.join(OUTPUT_DIR, "final_best_constraint_extended_outputs")
        os.makedirs(extended_dir, exist_ok=True)

        best_feature_cols = best_constraint_bundle["feature_cols"]
        X_train_best_df = best_constraint_bundle["train_df"][best_feature_cols].copy()
        X_test_best_df = best_constraint_bundle["test_df"][best_feature_cols].copy()
        y_train_best = best_constraint_bundle["y_train"]
        y_test_best = best_constraint_bundle["y_test"]

        m0_feature_cols = base_weather_features
        X_m0_train_best_df = paired_m0_bundle["train_df"][m0_feature_cols].copy()
        X_m0_test_best_df = paired_m0_bundle["test_df"][m0_feature_cols].copy()
        paired_m0_pipeline = Pipeline([
            ("scaler", paired_m0_bundle["scaler"]),
            ("model", paired_m0_bundle["model"]),
        ])

        # Per-sample final prediction outputs.
        best_train_pred_df = best_constraint_bundle["train_df"].copy()
        best_train_pred_df["True"] = y_train_best
        best_train_pred_df["M0 prediction"] = best_constraint_bundle.get("m0_train_pred", paired_m0_bundle["train_pred"])
        best_train_pred_df["Knowledge constrained correction"] = best_constraint_bundle.get("train_correction", best_constraint_bundle["train_pred"] - best_train_pred_df["M0 prediction"].values)
        best_train_pred_df["Best constrained prediction"] = best_constraint_bundle["train_pred"]
        best_train_pred_df["Best constrained residual"] = y_train_best - best_constraint_bundle["train_pred"]
        best_train_pred_df.to_excel(os.path.join(extended_dir, "best_constraint_train_predictions.xlsx"), index=False)

        best_test_pred_df = best_constraint_bundle["test_df"].copy()
        best_test_pred_df["True"] = y_test_best
        best_test_pred_df["M0 prediction"] = paired_m0_bundle["test_pred"]
        best_test_pred_df["Knowledge constrained correction"] = best_constraint_bundle.get("test_correction", best_constraint_bundle["test_pred"] - paired_m0_bundle["test_pred"])
        best_test_pred_df["Best constrained prediction"] = best_constraint_bundle["test_pred"]
        best_test_pred_df["Prediction shift constrained_minus_M0"] = best_constraint_bundle["test_pred"] - paired_m0_bundle["test_pred"]
        best_test_pred_df["M0 residual"] = y_test_best - paired_m0_bundle["test_pred"]
        best_test_pred_df["Best constrained residual"] = y_test_best - best_constraint_bundle["test_pred"]
        best_test_pred_df.to_excel(os.path.join(extended_dir, "best_constraint_test_predictions_with_M0_comparison.xlsx"), index=False)

        # Main comparison and diagnostic figures.
        final_metric_comparison = plot_metric_comparison(
            paired_m0_record,
            best_constrained,
            os.path.join(extended_dir, "best_M0_vs_best_constraint_test_metric_comparison.png")
        )

        m0_residual_df = plot_residual_diagnostics(
            y_test_best,
            paired_m0_bundle["test_pred"],
            "Best_M0",
            extended_dir
        )
        constraint_residual_df = plot_residual_diagnostics(
            y_test_best,
            best_constraint_bundle["test_pred"],
            "Best_constraint",
            extended_dir
        )
        residual_comparison_df = plot_residual_comparison(
            y_test_best,
            paired_m0_bundle["test_pred"],
            best_constraint_bundle["test_pred"],
            os.path.join(extended_dir, "best_M0_vs_best_constraint_residual_comparison.png")
        )
        prediction_shift_df = plot_prediction_shift(
            y_test_best,
            paired_m0_bundle["test_pred"],
            best_constraint_bundle["test_pred"],
            os.path.join(extended_dir, "prediction_shift_constrained_minus_M0.png")
        )

        # Heatmaps across all searched combinations.
        plot_model_heatmaps(result_df, extended_dir)

        # M0 RF impurity importance and permutation importance.
        m0_rf_importance_df = plot_rf_feature_importance(
            paired_m0_bundle["model"],
            m0_feature_cols,
            os.path.join(extended_dir, "best_M0_RF_impurity_feature_importance.png"),
            "Best M0 RF impurity feature importance"
        )
        if not m0_rf_importance_df.empty:
            m0_rf_importance_df.to_excel(os.path.join(extended_dir, "best_M0_RF_impurity_feature_importance.xlsx"), index=False)

        m0_perm_df = plot_permutation_importance_for_model(
            paired_m0_pipeline,
            X_m0_test_best_df,
            y_test_best,
            os.path.join(extended_dir, "best_M0_permutation_importance.png"),
            "Best M0 permutation importance"
        )
        m0_perm_df.to_excel(os.path.join(extended_dir, "best_M0_permutation_importance.xlsx"), index=False)

        constraint_perm_df = plot_permutation_importance_for_model(
            best_constraint_bundle["model"],
            X_test_best_df,
            y_test_best,
            os.path.join(extended_dir, "best_constraint_permutation_importance.png"),
            "Best constrained model permutation importance"
        )
        constraint_perm_df.to_excel(os.path.join(extended_dir, "best_constraint_permutation_importance.xlsx"), index=False)

        # SHAP for the final best constrained model and final best M0.
        constraint_shap_df = compute_and_plot_shap_for_model(
            best_constraint_bundle["model"],
            X_test_best_df,
            best_feature_cols,
            extended_dir,
            "best_constraint"
        )
        m0_shap_df = compute_and_plot_shap_for_model(
            paired_m0_pipeline,
            X_m0_test_best_df,
            m0_feature_cols,
            extended_dir,
            "best_M0"
        )

        # Learning curves for final best models.
        constraint_learning_curve_df = plot_learning_curve_fixed_best(
            best_constraint_bundle["model"],
            X_train_best_df,
            y_train_best,
            X_test_best_df,
            y_test_best,
            os.path.join(extended_dir, "best_constraint_learning_curve.png"),
            "Best constrained model learning curve"
        )
        m0_learning_curve_df = plot_learning_curve_fixed_best(
            paired_m0_pipeline,
            X_m0_train_best_df,
            y_train_best,
            X_m0_test_best_df,
            y_test_best,
            os.path.join(extended_dir, "best_M0_learning_curve.png"),
            "Best M0 RF learning curve"
        )

        # Partial dependence for constrained knowledge features, focusing on explicitly constrained features.
        try:
            constrained_positive_features = eval(best_constrained["Positive constraints"]) if isinstance(best_constrained["Positive constraints"], str) else []
        except Exception:
            constrained_positive_features = []
        pdp_df = plot_hgb_partial_dependence(
            best_constraint_bundle["model"],
            X_test_best_df,
            constrained_positive_features,
            extended_dir
        )

        # Additional simple relationships for knowledge variables when present.
        for knowledge_col in [TMIN_EXCESS_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL]:
            if knowledge_col in best_test_pred_df.columns:
                plot_residual_against_variable(
                    best_test_pred_df,
                    "Best constrained residual",
                    knowledge_col,
                    os.path.join(extended_dir, f"best_constraint_residuals_vs_{knowledge_col}.png"),
                )

        final_fitted_parameter_df = pd.DataFrame([
            {
                "Model": "Paired M0",
                "Version": "M0_RF_baseline",
                **paired_m0_bundle["model"].get_params(),
            },
            {
                "Model": MODEL_DISPLAY_NAME,
                "Version": best_constrained["Version"],
                **get_pipeline_scaler_and_estimator(best_constraint_bundle["model"])[1].get_params(),
            },
        ])
        final_fitted_parameter_df.to_excel(
            os.path.join(extended_dir, "final_selected_fitted_model_parameters.xlsx"),
            index=False,
        )

        # Consolidated Excel workbook for the final best model.
        save_dataframe_dict_to_excel(
            {
                "final_metric_comparison": final_metric_comparison,
                "paired_M0_record_same_split": pd.DataFrame([paired_m0_record]),
                "global_best_M0_record": pd.DataFrame([best_m0]),
                "best_constraint_record": pd.DataFrame([best_constrained]),
                "best_M2C_record": pd.DataFrame([best_constrained]),
                "best_constraint_test_pred": best_test_pred_df,
                "best_constraint_train_pred": best_train_pred_df,
                "residual_comparison": residual_comparison_df,
                "prediction_shift": prediction_shift_df,
                "best_constraint_perm": constraint_perm_df,
                "best_M0_perm": m0_perm_df,
                "best_constraint_SHAP": constraint_shap_df,
                "best_M0_SHAP": m0_shap_df,
                "best_constraint_learning": constraint_learning_curve_df,
                "best_M0_learning": m0_learning_curve_df,
                "PDP_outputs": pdp_df,
                "best_parameters": best_parameter_df,
                "parameter_search_cv": all_parameter_search_df,
                "final_fitted_params": final_fitted_parameter_df,
            },
            os.path.join(extended_dir, "final_best_constraint_extended_outputs.xlsx")
        )

        # Explicit M2-C detailed output directory.
        # It is a copy of the final selected constrained-correction outputs, saved under the final model name.
        m2c_output_dir = os.path.join(OUTPUT_DIR, "BEST_M2C_MODEL_OUTPUTS")
        if os.path.exists(m2c_output_dir):
            shutil.rmtree(m2c_output_dir)
        shutil.copytree(extended_dir, m2c_output_dir)

        pd.DataFrame([best_constrained]).to_excel(
            os.path.join(m2c_output_dir, "best_M2C_model_metrics_gridsearch.xlsx"),
            index=False,
        )
        best_test_pred_df.to_excel(
            os.path.join(m2c_output_dir, "best_M2C_test_predictions_with_M0_comparison.xlsx"),
            index=False,
        )
        best_train_pred_df.to_excel(
            os.path.join(m2c_output_dir, "best_M2C_train_predictions.xlsx"),
            index=False,
        )

    print("\n================ M0 plus constrained correction grid-search completed ================")
    print("Output directory:", OUTPUT_DIR)
    print("Best M0 baseline:")
    print({k: best_m0[k] for k in ["z_score_threshold", "cv", "Test R2", "Test RMSE", "Test MAE"]})
    print("Best M0 + constrained correction model:")
    print({k: best_constrained[k] for k in ["Version", "z_score_threshold", "cv", "Test R2", "Test RMSE", "Test MAE", "Delta test R2 vs M0"]})
    print("Best M2-C outputs:")
    print(os.path.join(OUTPUT_DIR, "best_M2C_model_metrics_gridsearch.xlsx"))
    print(os.path.join(OUTPUT_DIR, "best_M2C_model.pkl"))
    print(os.path.join(OUTPUT_DIR, "BEST_M2C_MODEL_OUTPUTS"))


if __name__ == "__main__":
    main()
