# -*- coding: utf-8 -*-
"""
M2 storage-protein knowledge-guided residual correction model:
R5 conditional residual constraint, grid-search version with aligned feature handling.

Main purpose
1. M0 is always the baseline model using only the original common weather factors.
2. TMIN_excess_20 and TMIN_above_20_flag are generated from TMIN, but they are not used as M0 inputs
   and are not used as protein-proxy inputs.
3. TMIN_excess_20 is used only for high/low TMIN grouping and for Protein_Heat_Risk construction.
4. Protein_Heat_Risk is used only in the residual constraint module.
5. The script loops over Z-score thresholds and CV folds, matching the ordinary residual-correction workflow:
      Z-score = [2, 3, 4]
      CV      = [4, 7, 10]
6. For each Z-score × CV combination, M0 and M2 are evaluated on the same cleaned samples and the same train/test split.
7. The final selected model is the combination with the highest M2 test R2, using M2 test RMSE as a tie-breaker.
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

warnings.filterwarnings("ignore")

# =========================================================
# 1. Paths and configuration
# =========================================================
ORIGINAL_DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
TEACHER_DATA_PATH = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\储藏蛋白-垩白-气象因子相关数据.xlsx"
OUTPUT_DIR = r"D:\实验\毕业论文\第四章\2.储藏蛋白知识引导模型构建\M2_R5_条件残差约束模型_网格筛选_特征对齐版"
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

# These variables are knowledge variables. They are not allowed to enter M0 or the protein-proxy input features.
KNOWLEDGE_ONLY_COLS = [TMIN_EXCESS_COL, TMIN_FLAG_COL, PREDICTED_PROTEIN_COL, PROTEIN_RISK_COL]

STUDENT_TEST_SIZE = 0.30
RANDOM_STATE = 42
TEACHER_Z_THRESHOLD = 4
TEACHER_N_SPLITS = 3
TEACHER_N_REPEATS = 10

STUDENT_Z_THRESHOLDS = [2, 3, 4]
STUDENT_CV_VALUES = [4, 7, 10]

PROTEIN_PROXY_N_ITER = 40
M0_N_ITER = 100

# Fixed residual-constraint parameters from previous R5 selection.
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

ALPHA_INFO_FIXED = {
    "alpha_mode": "m0",
    "alpha_base": 0.01,
    "alpha_amp": 0.10,
    "alpha_slope": 4.0,
    "alpha_center_quantile": 0.70,
    "alpha_direction": "increasing",
    "alpha_max_clip": 1.50,
}

# M0 parameter grid, aligned with the ordinary residual model workflow.
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

# =========================================================
# 2. Serializable helper models
# =========================================================
class ScaledRFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler
        self.model = model

    def fit(self, X, y):
        if self.scaler is None:
            self.scaler = StandardScaler()
        if self.model is None:
            self.model = RandomForestRegressor(random_state=RANDOM_STATE)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y=None):
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
            raise ValueError(f"Missing {self.threshold_col} in residual input features.")
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
    if alpha_info.get("alpha_direction", "increasing") == "decreasing":
        x = -x
    smooth = stable_sigmoid(alpha_info["alpha_slope"] * x)
    alpha_vec = alpha_info["alpha_base"] + alpha_info["alpha_amp"] * smooth
    return np.clip(alpha_vec, 0.0, alpha_info["alpha_max_clip"])


def tune_rf_repeated_cv(df, feature_cols, target_col, param_distributions, n_splits, n_repeats, n_iter, scoring="r2"):
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
    rs.fit(X_scaled, y)
    return rs.best_estimator_, scaler, rs


def generate_oof_prediction_rf(df, feature_cols, target_col, best_params, n_splits=3):
    X = df[feature_cols].values
    y = df[target_col].values
    oof_pred = np.zeros(len(df), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, valid_idx in kf.split(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_valid_scaled = scaler.transform(X[valid_idx])
        model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
        model.fit(X_train_scaled, y[train_idx])
        oof_pred[valid_idx] = model.predict(X_valid_scaled)
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


def train_fixed_scaled_rf(X, y, params):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(random_state=RANDOM_STATE, **params)
    model.fit(X_scaled, y)
    return ScaledRFRegressor(scaler=scaler, model=model)


def build_residual_feature_df(df, weather_cols, protein_col):
    """
    Build residual-module input features.

    Important feature handling:
    1. TMIN_excess_20 is NOT used in M0 or the protein-proxy model.
    2. Here it is retained only as a residual-module mechanism column for high/low TMIN grouping.
    3. TMIN_excess_20 is excluded from the low/high residual regressors themselves.
    4. Protein_Heat_Risk is used as the constrained mechanism feature in the high-TMIN residual model.
    """
    out = df[weather_cols].copy()
    if TMIN_EXCESS_COL not in df.columns:
        raise ValueError(f"Missing {TMIN_EXCESS_COL}; cannot build R5 residual features.")
    out[PROTEIN_COL] = df[protein_col].values
    out[TMIN_EXCESS_COL] = df[TMIN_EXCESS_COL].values
    out[PROTEIN_RISK_COL] = df[protein_col].values * df[TMIN_EXCESS_COL].values
    return out


def build_monotonic_constraints(feature_cols, positive_features):
    return [1 if col in positive_features else 0 for col in feature_cols]


def prepare_teacher_residual_df(teacher_df_clean, weather_cols, m0_model, m0_scaler):
    teacher_m0_pred = m0_model.predict(m0_scaler.transform(teacher_df_clean[weather_cols].values))
    residual_target = teacher_df_clean[TARGET_COL].values - teacher_m0_pred
    out = teacher_df_clean.copy()
    out["M0 prediction on teacher"] = teacher_m0_pred
    out["M0 residual on teacher"] = residual_target
    out = add_protein_heat_risk(out, PROTEIN_COL)
    return out


def train_r5_conditional_residual_model(residual_train_df, weather_cols):
    feature_df = build_residual_feature_df(residual_train_df, weather_cols, PROTEIN_COL)
    input_feature_cols = feature_df.columns.tolist()

    # TMIN_excess_20 is needed by the conditional wrapper for grouping,
    # but it is not used as a fitted residual-regressor feature.
    model_feature_cols = [c for c in input_feature_cols if c != TMIN_EXCESS_COL]

    model_df = feature_df.copy()
    model_df["M0 residual on teacher"] = residual_train_df["M0 residual on teacher"].values

    high_mask = model_df[TMIN_EXCESS_COL].values > 0
    low_df = model_df.loc[~high_mask].copy()
    high_df = model_df.loc[high_mask].copy()

    if len(low_df) >= 6:
        low_model = train_fixed_scaled_rf(
            low_df[model_feature_cols].values,
            low_df["M0 residual on teacher"].values,
            LOW_RESIDUAL_RF_PARAMS,
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
        high_model.fit(high_df[model_feature_cols].values, high_df["M0 residual on teacher"].values)
    else:
        high_model = ConstantRegressor(float(high_df["M0 residual on teacher"].mean()) if len(high_df) else 0.0)

    model = ConditionalResidualModel(
        input_feature_cols=input_feature_cols,
        model_feature_cols=model_feature_cols,
        threshold_col=TMIN_EXCESS_COL,
        threshold_value=0.0,
        low_model=low_model,
        high_model=high_model,
    )
    info = {
        "Residual model": "R5_conditional_highTMIN_risk_constraint",
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
    }
    return model, info, input_feature_cols

# =========================================================
# 4. Main workflow
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

    # Aligned feature handling: threshold/risk/predicted-protein variables are excluded from M0 and protein proxy.
    excluded_teacher = set([TARGET_COL, PROTEIN_COL] + KNOWLEDGE_ONLY_COLS)
    excluded_original = set([original_target_col] + KNOWLEDGE_ONLY_COLS)
    teacher_weather_cols = [c for c in teacher_numeric.columns if c not in excluded_teacher]
    original_weather_cols = [c for c in original_numeric.columns if c not in excluded_original]
    base_weather_features = [c for c in original_weather_cols if c in teacher_weather_cols]

    if len(base_weather_features) == 0:
        raise ValueError("No common numeric weather features were found.")
    if TMIN_COL not in base_weather_features:
        raise ValueError(f"{TMIN_COL} must be in base weather features to construct threshold variables.")

    print("\nBase weather features for M0 and protein proxy:")
    print(base_weather_features)
    pd.DataFrame({"base_weather_features": base_weather_features}).to_excel(
        os.path.join(OUTPUT_DIR, "base_weather_features.xlsx"), index=False
    )

    # Teacher data: base features + threshold variables retained only for grouping/risk construction.
    teacher_df = teacher_numeric[[TARGET_COL, PROTEIN_COL] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    teacher_clean, teacher_outliers = remove_outliers_by_zscore(
        teacher_df, base_weather_features + [PROTEIN_COL], TEACHER_Z_THRESHOLD
    )
    print("\nTeacher valid n:", len(teacher_df))
    print("Teacher cleaned n:", len(teacher_clean))
    print("Teacher outliers removed:", len(teacher_outliers))

    # Protein proxy is trained only from base weather features.
    teacher_cv = RepeatedKFold(n_splits=TEACHER_N_SPLITS, n_repeats=TEACHER_N_REPEATS, random_state=RANDOM_STATE)
    protein_model, protein_scaler, protein_search = tune_rf_repeated_cv(
        teacher_clean,
        base_weather_features,
        PROTEIN_COL,
        protein_proxy_param_grid,
        n_splits=TEACHER_N_SPLITS,
        n_repeats=TEACHER_N_REPEATS,
        n_iter=PROTEIN_PROXY_N_ITER,
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

    # Student data: base weather inputs, threshold variables for risk construction, predicted protein for residual model.
    student_df = original_numeric[[original_target_col] + base_weather_features + [TMIN_EXCESS_COL, TMIN_FLAG_COL]].dropna().copy()
    student_df[PREDICTED_PROTEIN_COL] = protein_model.predict(protein_scaler.transform(student_df[base_weather_features].values))
    student_df = add_protein_heat_risk(student_df, PREDICTED_PROTEIN_COL)
    print("\nStudent valid n:", len(student_df))

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
            print(f"\nTraining R5 grid combination: Z-score={z_thr}, CV={cv}")

            m0_model, m0_scaler, m0_search, m0_train_pred, m0_test_pred = train_m0_rf_on_fixed_split(
                X_train, y_train, X_test, cv
            )
            m0_train_metrics = calculate_metrics(y_train, m0_train_pred, "M0 train")
            m0_test_metrics = calculate_metrics(y_test, m0_test_pred, "M0 test")

            residual_train_df = prepare_teacher_residual_df(teacher_clean, base_weather_features, m0_model, m0_scaler)
            residual_model, residual_info, residual_feature_cols = train_r5_conditional_residual_model(
                residual_train_df, base_weather_features
            )

            train_features = build_residual_feature_df(train_df, base_weather_features, PREDICTED_PROTEIN_COL)[residual_feature_cols]
            test_features = build_residual_feature_df(test_df, base_weather_features, PREDICTED_PROTEIN_COL)[residual_feature_cols]
            raw_train_correction = residual_model.predict(train_features.values)
            raw_test_correction = residual_model.predict(test_features.values)

            alpha_info = ALPHA_INFO_FIXED.copy()
            alpha_info["alpha_center"] = float(np.quantile(m0_test_pred, alpha_info["alpha_center_quantile"]))
            alpha_info["alpha_scale"] = float(np.std(m0_test_pred))
            if alpha_info["alpha_scale"] <= 1e-12:
                alpha_info["alpha_scale"] = 1.0

            train_alpha = compute_smooth_alpha(m0_train_pred, alpha_info)
            test_alpha = compute_smooth_alpha(m0_test_pred, alpha_info)
            m2_train_pred = m0_train_pred + train_alpha * raw_train_correction
            m2_test_pred = m0_test_pred + test_alpha * raw_test_correction

            m2_train_metrics = calculate_metrics(y_train, m2_train_pred, "M2 train")
            m2_test_metrics = calculate_metrics(y_test, m2_test_pred, "M2 test")

            rec = {
                "Version": "M2_R5_conditional_residual_constraint",
                "z_score_threshold": z_thr,
                "cv": cv,
                "n_samples_after_cleaning": len(student_clean),
                "n_outliers_removed": len(student_outliers),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "M0_best_params": json.dumps(m0_search.best_params_, ensure_ascii=False),
                "M0_cv_r2_on_train": float(m0_search.best_score_),
                "Protein_proxy_best_params": json.dumps(protein_search.best_params_, ensure_ascii=False),
                "Protein_proxy_cv_R2": float(protein_search.best_score_),
                "Protein_proxy_oof_R2": protein_oof_metrics["Protein OOF prediction R2"],
                "Alpha params": json.dumps(alpha_info, ensure_ascii=False),
                "Train alpha min": float(np.min(train_alpha)),
                "Train alpha max": float(np.max(train_alpha)),
                "Train alpha mean": float(np.mean(train_alpha)),
                "Test alpha min": float(np.min(test_alpha)),
                "Test alpha max": float(np.max(test_alpha)),
                "Test alpha mean": float(np.mean(test_alpha)),
            }
            rec.update(residual_info)
            rec.update(m0_train_metrics)
            rec.update(m0_test_metrics)
            rec.update(m2_train_metrics)
            rec.update(m2_test_metrics)
            rec.update({
                "Delta test R2 M2_minus_M0": rec["M2 test R2"] - rec["M0 test R2"],
                "Delta test RMSE M2_minus_M0": rec["M2 test RMSE"] - rec["M0 test RMSE"],
                "Delta test MAE M2_minus_M0": rec["M2 test MAE"] - rec["M0 test MAE"],
                "M0 overfit gap": rec["M0 train R2"] - rec["M0 test R2"],
                "M2 overfit gap": rec["M2 train R2"] - rec["M2 test R2"],
            })
            records.append(rec)

            is_better = best_record is None or (
                rec["M2 test R2"] > best_record["M2 test R2"] or
                (np.isclose(rec["M2 test R2"], best_record["M2 test R2"]) and rec["M2 test RMSE"] < best_record["M2 test RMSE"])
            )
            if is_better:
                best_record = rec.copy()
                best_bundle = {
                    "student_clean": student_clean.copy(),
                    "train_df": train_df.copy(),
                    "test_df": test_df.copy(),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "m0_model": m0_model,
                    "m0_scaler": m0_scaler,
                    "residual_model": residual_model,
                    "residual_feature_cols": residual_feature_cols,
                    "alpha_info": alpha_info,
                    "m0_train_pred": m0_train_pred,
                    "m0_test_pred": m0_test_pred,
                    "raw_train_correction": raw_train_correction,
                    "raw_test_correction": raw_test_correction,
                    "train_alpha": train_alpha,
                    "test_alpha": test_alpha,
                    "m2_train_pred": m2_train_pred,
                    "m2_test_pred": m2_test_pred,
                    "y_train": y_train,
                    "y_test": y_test,
                }

    if best_record is None:
        raise RuntimeError("No valid R5 model was trained.")

    result_df = pd.DataFrame(records)
    result_df.to_excel(os.path.join(OUTPUT_DIR, "R5_conditional_residual_constraint_all_results_gridsearch.xlsx"), index=False)
    core_cols = [
        "Version", "z_score_threshold", "cv", "M0 test R2", "M0 test RMSE", "M0 test MAE",
        "M2 test R2", "M2 test RMSE", "M2 test MAE",
        "Delta test R2 M2_minus_M0", "Delta test RMSE M2_minus_M0", "Delta test MAE M2_minus_M0",
        "M0_cv_r2_on_train", "M0_best_params", "Protein_proxy_cv_R2", "Protein_proxy_oof_R2",
        "Low group n", "High group n", "Residual input feature cols", "Residual model feature cols",
    ]
    # Robust core export: keep only existing columns to avoid failure if a metadata column name changes.
    core_cols_existing = [c for c in core_cols if c in result_df.columns]
    missing_core_cols = [c for c in core_cols if c not in result_df.columns]
    if missing_core_cols:
        print("Warning: missing core metric columns skipped:", missing_core_cols)
    result_df[core_cols_existing].to_excel(os.path.join(OUTPUT_DIR, "R5_conditional_residual_constraint_core_metrics_gridsearch.xlsx"), index=False)
    pd.DataFrame([best_record]).to_excel(os.path.join(OUTPUT_DIR, "best_R5_conditional_residual_constraint_metrics_gridsearch.xlsx"), index=False)

    # Save final best models and predictions.
    joblib.dump(protein_model, os.path.join(OUTPUT_DIR, "protein_proxy_model.pkl"))
    joblib.dump(protein_scaler, os.path.join(OUTPUT_DIR, "protein_proxy_scaler.pkl"))
    joblib.dump(best_bundle["m0_model"], os.path.join(OUTPUT_DIR, "best_M0_model.pkl"))
    joblib.dump(best_bundle["m0_scaler"], os.path.join(OUTPUT_DIR, "best_M0_scaler.pkl"))
    joblib.dump(best_bundle["residual_model"], os.path.join(OUTPUT_DIR, "best_R5_conditional_residual_model.pkl"))
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
    train_pred_df.to_excel(os.path.join(OUTPUT_DIR, "best_R5_train_predictions.xlsx"), index=False)
    test_pred_df.to_excel(os.path.join(OUTPUT_DIR, "best_R5_test_predictions.xlsx"), index=False)

    print("\n================ R5 conditional residual constraint grid search completed ================")
    print("Output directory:", OUTPUT_DIR)
    print("Best z-score:", best_record["z_score_threshold"])
    print("Best CV:", best_record["cv"])
    print("Best M0 test R2:", best_record["M0 test R2"])
    print("Best M2 test R2:", best_record["M2 test R2"])
    print("Delta test R2 M2_minus_M0:", best_record["Delta test R2 M2_minus_M0"])


if __name__ == "__main__":
    main()
