import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import shap

# 图表参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 600

# 1. 读取数据
# 所有数据为数值格式，不能有数字外的其他字符出现，请严格检查
data_path = r"C:\Users\27144\Desktop\20240506气候\中\中1.xlsx"
data = pd.read_excel(data_path)

# 2. 将第1列与其他所有列的数据进行建模
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
# 设置离群值剔除的Z-score阈值范围
z_score_thresholds = [4.0] #2.5，3，3.5，4
best_r2 = float('-inf')
best_rmse = float('inf')
best_params = None
best_forest = None
best_z_score_threshold = None
best_cv = None
# 遍历不同的Z-score阈值和交叉验证折数
for z_score_threshold in z_score_thresholds:
    for cv in [5]: #5，7，10，15
        # 3. 清洗数据噪声，根据当前阈值删除离群值
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outliers = np.where(np.any(z_scores >= z_score_threshold, axis=1))[0]
        X_cleaned = np.delete(X, outliers, axis=0)
        y_cleaned = np.delete(y, outliers, axis=0)
        # 特征选择：去除高度相关的特征
        X_df = pd.DataFrame(X_cleaned)
        correlation_matrix = X_df.corr().abs()
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.5)] #自行选择
        features_to_keep = [col for col in upper_tri.columns if col not in to_drop or upper_tri[col].max() == 1]
        X_filtered = X_cleaned[:, features_to_keep]

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_cleaned, test_size=0.30, random_state=42)

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 定义随机森林回归模型及参数网格搜索
        forest_reg = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100], #50，100，150，200
            'max_depth': [10], #None, 5，10, 20
            'min_samples_split': [5], #2，5，7，10
            'min_samples_leaf': [1], #1, 2, 3, 4，5
            'max_features': ['sqrt']
        }

        grid_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_grid,
            n_iter=100,
            cv=cv,
            scoring='r2',
            refit=True,
            random_state=42,
            n_jobs=-1
        )

        grid_search.fit(X_train_scaled, y_train)

        # 输出最佳参数并使用最优参数训练模型
        print(f"Best parameters for Z-score threshold {z_score_threshold} and CV {cv}: {grid_search.best_params_}")
        best_forest = grid_search.best_estimator_

        # 计算并输出训练集与测试集上的R2分数
        train_r2 = r2_score(y_train, best_forest.predict(X_train_scaled))
        test_r2 = r2_score(y_test, best_forest.predict(X_test_scaled))
        print(f"Train R2 Score: {train_r2:.4f}, Test R2 Score: {test_r2:.4f}")

        # 计算并输出RMSE
        rmse = np.sqrt(mean_squared_error(y_test, best_forest.predict(X_test_scaled)))
        print(f"Test RMSE: {rmse:.4f}")

        # 更新最佳模型参数和性能指标（如果当前模型的表现更好）
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_rmse = rmse
            best_params = grid_search.best_params_
            best_z_score_threshold = z_score_threshold
            best_cv = cv
# 输出最终的最佳结果
print("\nFinal Best R2 Score (Test): ", best_r2)
print("Final Best RMSE: ", best_rmse)
print("Final Best Parameters: ", best_params)
print("Final Best Z-Score Threshold: ", best_z_score_threshold)
print("Final Best CV: ", best_cv)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 绘制最佳模型的实际值与预测值的对比图
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(10, 6))

# 使用最佳模型进行预测
best_y_pred = best_forest.predict(X_test_scaled)

ax.scatter(y_test, best_y_pred, s=80, alpha=0.8, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Fit')  # 添加最佳拟合线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression Model')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend()  # 显示图例
plt.savefig("rf_regression_model_plot.png", dpi=300, bbox_inches="tight")
plt.show()
# Test R2 Score和Test RMSE为最后所需结果
# Train R2 Score接近1则表示模型过拟合，结果不可用

#shap
# 筛选特征后，同时记录下被保留特征的原始名字
feature_names_kept = data.columns[1:][features_to_keep]

# 提取特征重要性及对应的特征名
feature_importances = best_forest.feature_importances_
# 确保特征名字与重要性值正确对应
feature_importance_series = pd.Series(feature_importances, index=feature_names_kept)

# 使用SHAP库解释最佳模型的预测
explainer = shap.Explainer(best_forest)
shap_values = explainer.shap_values(X_test_scaled)

plot_size = (10, 8) #修改图的比例
max_display = 10  #修改图中显示的特征数量
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names_kept, plot_size=plot_size, max_display=max_display)
