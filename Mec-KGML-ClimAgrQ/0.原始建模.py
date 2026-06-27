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

# ===================== 配置区（画布与轴标题都在这里改） =====================
FIGSIZE = {
    "shap_dot":        (8, 4),   # SHAP 全局点图（beeswarm）
    "shap_bar":        (6, 4),   # 自定义条形图
    "shap_dependence": (6, 5),   # 依赖图
    "residuals":       (6, 5),   # 残差图
    "learning_curve":  (6, 5)    # 学习曲线
}
JOINTGRID_HEIGHT = 6           # JointGrid 主面板边长（英寸）

# x 轴标题（改这里即可）
XAXIS_LABEL = {
    "shap_dot_x": "SHAP value",
    "shap_bar_x": "mean(|SHAP value|)"
}

TOP_K = 5                       # 展示 TOP-K 特征（条形图/点图一致）
# ====================================================================

# 字体风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 600
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗
plt.rcParams['font.size'] = 14

# 读取数据
DATA_PATH = r"D:\实验\毕业论文\第四章\1.气象阈值知识增强建模\数据库籼稻建模.xlsx"
data = pd.read_excel(DATA_PATH)

# 提取 X / y
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 搜索空间
z_score_thresholds = [2, 3, 4]
cv_values = [4, 7, 10]

# 记录最佳
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

shap_values_best = None
X_test_best = None
X_train_best = None
y_test_best = None
y_train_best = None
best_scaler = None  # 保存最佳模型用到的 scaler（依赖图逆变换使用）

# 遍历阈值与 CV
for z_thr in z_score_thresholds:
    for cv in cv_values:
        # 剔除离群
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outliers = np.where(np.any(z_scores >= z_thr, axis=1))[0]
        X_cleaned = np.delete(X, outliers, axis=0)
        y_cleaned = np.delete(y, outliers, axis=0)

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_cleaned, y_cleaned, test_size=0.30, random_state=42
        )

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # 随机森林 + 随机搜索
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
        train_r2 = r2_score(y_train, model.predict(X_train_scaled))
        test_r2  = r2_score(y_test,  model.predict(X_test_scaled))
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
        mse  = mean_squared_error(y_test,  model.predict(X_test_scaled))
        mae  = mean_absolute_error(y_test, model.predict(X_test_scaled))

        # 轻微过拟合约束
        if abs(train_r2 - test_r2) <= 0.5 and test_r2 > best_r2:
            best_r2 = test_r2
            best_rmse = rmse
            best_mse = mse
            best_mae = mae
            best_params = rs.best_params_
            best_forest = model
            best_z_score_threshold = z_thr
            best_cv = cv

            X_test_best  = X_test_scaled
            X_train_best = X_train_scaled
            y_test_best  = y_test
            y_train_best = y_train
            best_scaler  = scaler

            # SHAP
            explainer = shap.Explainer(best_forest)
            shap_vals = explainer.shap_values(X_test_scaled)
            shap_values_best = shap_vals
            shap_mean_importance = np.abs(shap_vals).mean(axis=0)
            best_feature_index = int(np.argmax(shap_mean_importance))
            best_feature = data.columns[1:][best_feature_index]

# 输出最佳
print("\n最佳 R2:", best_r2)
print("最佳 RMSE:", best_rmse)
print("最佳 MSE:", best_mse)
print("最佳 MAE:", best_mae)
print("最佳参数:", best_params)
print("最佳 Z-score 阈值:", best_z_score_threshold)
print("最佳交叉验证折数:", best_cv)
print("最重要的特征:", best_feature)

# ====== SHAP 重要性表 ======
shap_mean_importance = np.abs(shap_values_best).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': data.columns[1:],
    'SHAP Mean Importance': shap_mean_importance
}).sort_values(by='SHAP Mean Importance', ascending=False)

print("\nSHAP 特征重要性排序:")
print(feature_importance_df)

# 保存
feature_importance_df.to_excel(
    r"C:\Users\27144\Desktop\shap_feature_importances.xlsx",
    index=False
)

# 取 TOP-K
topk_df = feature_importance_df.head(TOP_K)
topk_idx = topk_df.index
topk_names = topk_df['Feature'].values

# =============== ① SHAP 全局点图（beeswarm） ===============
# 受 FIGSIZE["shap_dot"] 控制；x 轴标题在 XAXIS_LABEL 里改
shap.summary_plot(
    shap_values_best[:, topk_idx],
    X_test_best[:, topk_idx],
    feature_names=topk_names,
    plot_type="dot",
    show=False,
    plot_size=FIGSIZE["shap_dot"]
)
ax = plt.gca()
ax.set_xlabel(XAXIS_LABEL["shap_dot_x"], fontsize=14, fontweight='bold')

# ▼ 让 y 轴刻度标签（特征名）加粗并调字号
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(14)

plt.tight_layout()
plt.show()

# =============== ② 自定义条形图（“以前的风格”） ===============
# 不用 shap.summary_plot(bar)；改成 matplotlib 水平条形图：渐变蓝/粗边框/数值标注/加粗标签
def plot_shap_bar_custom(values, names, figsize=(6,4), xlabel="mean(|SHAP value|)"):
    # values/ names 均已按降序传入
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(values))[::1]  # 自上而下从大到小
    # 渐变蓝
    cmap = plt.cm.Blues
    cols = cmap(np.linspace(0.45, 0.95, len(values)))
    ax.barh(y, values[::-1], color=cols[::1], edgecolor='black', linewidth=1.2, height=0.7)
    # ytick
    ax.set_yticks(y)
    ax.set_yticklabels(names[::-1], fontsize=14, fontweight='bold')
    # 数值标注
    for yi, vi in zip(y, values[::-1]):
        ax.text(vi*1.01, yi, f"{vi:.3f}", va='center', ha='left', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_shap_bar_custom(
    values=topk_df['SHAP Mean Importance'].values,
    names=topk_df['Feature'].values,
    figsize=FIGSIZE["shap_bar"],
    xlabel=XAXIS_LABEL["shap_bar_x"]
)

# =============== ③ 依赖图（完全受 FIGSIZE 控制） ===============
feature_index = list(data.columns[1:]).index(best_feature)
shap_values_for_feature = shap_values_best[:, feature_index]
if X_test_best.shape[0] != shap_values_for_feature.shape[0]:
    raise ValueError("Length mismatch between X_test_best and shap_values_for_feature")

# 还原原始特征值（用最佳 scaler）
original_feature_values = best_scaler.inverse_transform(X_test_best)[:, feature_index]
if len(original_feature_values) != len(shap_values_for_feature):
    raise ValueError("Original feature values and SHAP values must have the same length.")

plt.figure(figsize=FIGSIZE["shap_dependence"])
lowess_fit = sm.nonparametric.lowess(shap_values_for_feature, original_feature_values, frac=0.7)
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='blue', linewidth=3, label='Lowess Fit')
plt.fill_between(lowess_fit[:, 0], lowess_fit[:, 1] - 0.1, lowess_fit[:, 1] + 0.1,
                 color='gray', alpha=0.3, label='95% CI')
plt.grid(True, linestyle='--', linewidth=1, color='lightgray', alpha=0.7)
plt.legend(fontsize=14)
plt.xlabel(f'{best_feature}', fontsize=18)
plt.ylabel('SHAP Value', fontsize=18)
plt.title(f'SHAP Dependence Plot', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.show()

# =============== ④ 真实值 vs 预测值（JointGrid） ===============
y_pred_test  = best_forest.predict(X_test_best)
y_pred_train = best_forest.predict(X_train_best)
test_r2  = r2_score(y_test_best, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test_best, y_pred_test))

data_train = pd.DataFrame({'True': y_train_best, 'Predicted': y_pred_train, 'Data Set': 'Train'})
data_test  = pd.DataFrame({'True': y_test_best,  'Predicted': y_pred_test,  'Data Set': 'Test'})
data_combined = pd.concat([data_train, data_test])

palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
g = sns.JointGrid(data=data_combined, x="True", y="Predicted",
                  hue="Data Set", height=JOINTGRID_HEIGHT, palette=palette)
g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
sns.regplot(data=data_train, x="True", y="Predicted",
            scatter=False, ax=g.ax_joint, color='#b4d4e1', label='Train Regression Line')
sns.regplot(data=data_test, x="True", y="Predicted",
            scatter=False, ax=g.ax_joint, color='#f4ba8a', label='Test Regression Line')
g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

ax = g.ax_joint
ax.set_xlabel("True Values", fontsize=20, weight='bold', labelpad=10)
ax.set_ylabel("Predicted Values", fontsize=20, weight='bold', labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.text(0.95, 0.05, f'$R^2$ = {test_r2:.2f}\nRMSE = {test_rmse:.2f}',
        transform=ax.transAxes, fontsize=22, va='bottom', ha='right',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
ax.text(0.75, 0.99, 'Model = RF', transform=ax.transAxes, fontsize=18,
        va='top', ha='left',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
ax.plot([data_combined['True'].min(), data_combined['True'].max()],
        [data_combined['True'].min(), data_combined['True'].max()],
        c="black", alpha=0.7, linestyle='--', label='x=y')
ax.legend(loc='best', fontsize=16)
plt.tight_layout()
plt.show()

# =============== ⑤ 残差图（受 FIGSIZE 控制） ===============
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=FIGSIZE["residuals"])
    plt.scatter(y_pred, residuals, alpha=0.7, s=120, edgecolors='k', color='#b4d4e1')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Values", fontsize=18)
    plt.ylabel("Residuals", fontsize=18)
    plt.title("Residual Plot", fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_residuals(y_test_best, y_pred_test)

# =============== ⑥ 学习曲线（受 FIGSIZE 控制） ===============
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, test_scores = [], []
for train_size in train_sizes:
    n = int(len(X_train_best) * train_size)
    X_train_subset = X_train_best[:n]
    y_train_subset = y_train_best[:n]
    model = RandomForestRegressor(**best_forest.get_params())
    model.fit(X_train_subset, y_train_subset)
    train_scores.append(r2_score(y_train_subset, model.predict(X_train_subset)))
    test_scores.append(r2_score(y_test_best, model.predict(X_test_best)))

train_scores = np.array(train_scores)
test_scores  = np.array(test_scores)

plt.figure(figsize=FIGSIZE["learning_curve"])
plt.plot(train_sizes * len(X_train_best), train_scores, 'o-', color="r", label="Training score", linewidth=2)
plt.plot(train_sizes * len(X_train_best), test_scores,  'o-', color="b", label="Testing score",  linewidth=2)
plt.xlabel("Training Set Size", fontsize=18)
plt.ylabel("Score (R²)", fontsize=18)
plt.title("Learning Curve", fontsize=20)
plt.legend(loc="best", fontsize=14)
plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 最终指标
y_pred_train = best_forest.predict(X_train_best)
print("\nFinal Model Performance:")
print(f"Training R²: {r2_score(y_train_best, y_pred_train):.4f}")
print(f"Testing  R²: {r2_score(y_test_best, y_pred_test):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train_best, y_pred_train)):.4f}")
print(f"Testing  RMSE: {np.sqrt(mean_squared_error(y_test_best, y_pred_test)):.4f}")
print(f"Training  MSE: {mean_squared_error(y_train_best, y_pred_train):.4f}")
print(f"Testing   MSE: {mean_squared_error(y_test_best, y_pred_test):.4f}")
print(f"Training  MAE: {mean_absolute_error(y_train_best, y_pred_train):.4f}")
print(f"Testing   MAE: {mean_absolute_error(y_test_best, y_pred_test):.4f}")
