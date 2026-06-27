import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import statsmodels.api as sm

# ===================== 配置 =====================
FIGSIZE = {
    "shap_dot": (8, 4),
    "shap_bar": (6, 4),
    "shap_dependence": (6, 5),
    "residuals": (6, 5),
    "learning_curve": (6, 5)
}

JOINTGRID_HEIGHT = 6
TOP_K = 5

XAXIS_LABEL = {
    "shap_dot_x": "SHAP value",
    "shap_bar_x": "mean(|SHAP value|)"
}

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 600
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 14

# ===================== 数据读取 =====================
DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
data = pd.read_excel(DATA_PATH)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
feature_names = data.columns[1:]

# ===================== 参数搜索 =====================
z_score_thresholds = [2, 3, 4]
cv_values = [4, 7, 10]

best_r2 = -np.inf
best_model = None
best_scaler = None
best_shap_values = None

best_X_train, best_X_test = None, None
best_y_train, best_y_test = None, None
best_feature = None

# ===================== 模型训练 =====================
for z_thr in z_score_thresholds:
    for cv in cv_values:

        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outliers = np.where(np.any(z_scores >= z_thr, axis=1))[0]

        X_clean = np.delete(X, outliers, axis=0)
        y_clean = np.delete(y, outliers, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [2, 5, 10, 20, 30],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 7],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['log2', 'sqrt']
        }

        rs = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=100,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        ).fit(X_train_s, y_train)

        best_est = rs.best_estimator_

        y_pred = best_est.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_model = best_est
            best_scaler = scaler

            best_X_train, best_X_test = X_train_s, X_test_s
            best_y_train, best_y_test = y_train, y_test

            explainer = shap.Explainer(best_est)
            best_shap_values = explainer(X_test_s)

# ===================== 基本评估 =====================
y_pred_test = best_model.predict(best_X_test)
y_pred_train = best_model.predict(best_X_train)

print("\nBest R2:", best_r2)
print("RMSE:", np.sqrt(mean_squared_error(best_y_test, y_pred_test)))
print("MAE:", mean_absolute_error(best_y_test, y_pred_test))

# ===================== SHAP =====================
shap_mean = np.abs(best_shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": shap_mean
}).sort_values("Importance", ascending=False)

print(importance_df.head(10))

top_features = importance_df.head(TOP_K)["Feature"].values
top_idx = [list(feature_names).index(f) for f in top_features]

shap.summary_plot(
    best_shap_values.values[:, top_idx],
    best_X_test[:, top_idx],
    feature_names=top_features,
    show=False
)
plt.show()

# ===================== 残差 =====================
residuals = best_y_test - y_pred_test

plt.figure(figsize=FIGSIZE["residuals"])
plt.scatter(y_pred_test, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()

# ===================== 学习曲线 =====================
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, test_scores = [], []

for s in train_sizes:
    n = int(len(best_X_train) * s)
    model = RandomForestRegressor(**best_model.get_params())
    model.fit(best_X_train[:n], best_y_train[:n])

    train_scores.append(r2_score(best_y_train[:n], model.predict(best_X_train[:n])))
    test_scores.append(r2_score(best_y_test, model.predict(best_X_test)))

plt.figure(figsize=FIGSIZE["learning_curve"])
plt.plot(train_sizes, train_scores)
plt.plot(train_sizes, test_scores)
plt.show()

# ===================== 独立样本预测（核心新增） =====================
INDEP_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\M0.xlsx"
indep = pd.read_excel(INDEP_PATH)

# 强制对齐特征列（关键步骤）
indep_X = indep.iloc[:, 1:]
indep_X = indep_X[feature_names]  # 保证列顺序一致

# 缺失值处理（防止报错）
indep_X = indep_X.fillna(indep_X.mean())

# 标准化
indep_X_s = best_scaler.transform(indep_X)

# 预测
indep_pred = best_model.predict(indep_X_s)

# 回写第一列
indep.iloc[:, 0] = indep_pred

# 输出
OUT_PATH = r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\M0_predicted.xlsx"
indep.to_excel(OUT_PATH, index=False)

print("\nIndependent prediction completed")
print("Saved to:", OUT_PATH)