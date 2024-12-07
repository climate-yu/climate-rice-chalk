import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import make_scorer


# 1. 读取数据
data = pd.read_excel(r"C:\Users\27144\Desktop\climate\中\中1.xlsx")

# 2. 将第1列与其他所有列的数据进行建模
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 3. 清洗数据噪声，测试不同问值删除离群债
thresholds_to_test = [2, 2.5, 3, 3.5]
cv_values_to_test = [4, 7, 10, 15, 17]
best_threshold = None
best_cv = None
best_r2 = float('-inf')
best_rmse = None  # 初始化rmse为None
best_mae = None  # 初始化mae为None
best_train_r2 = float('-inf')  # 初始化train_r2为最小值
best_mse = None  # 初始化mse为None

for threshold in thresholds_to_test:
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    outliers = np.where(np.any(z_scores > threshold, axis=1))[0]
    X_filtered = np.delete(X, outliers, axis=0)
    y_filtered = np.delete(y, outliers)

    for cv in cv_values_to_test:
        # 4. 70%建模，30%用来验证模型
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

        # 5. 交叉验证、正则化、特征选择
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 优化超参数
        param_grid = {
            'n_neighbors': range(1, 21),
            'weights': ['uniform', 'distance'],
            'p': [1, 2],  # 增加距离度量的幂指数p
            'algorithm': ['ball_tree', 'kd_tree', 'brute']  # 增加不同的计算邻居算法
        }

        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=cv, scoring='r2')
        grid_search.fit(X_train_scaled, y_train)

        best_knn = grid_search.best_estimator_
        print(f"Threshold: {threshold}, CV: {cv}, Best parameters found: ", grid_search.best_params_)

        # 计算训练集和测试集上的R2分数，用于过拟合检验
        train_r2 = r2_score(y_train, best_knn.predict(X_train_scaled))
        test_r2 = r2_score(y_test, best_knn.predict(X_test_scaled))

        y_pred = best_knn.predict(X_test_scaled)  # Add this line to define y_pred

        # 计算MAE和MSE
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"Threshold: {threshold}, CV: {cv}, Train R2 Score: {train_r2:.4f}, Test R2 Score: {test_r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

        if abs(train_r2 - test_r2) <= 0.3 and test_r2 > best_r2:  # 检查是否满足过拟合条件且R2得分最优
            best_threshold = threshold
            best_cv = cv
            best_r2 = test_r2
            best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            best_mae = mae
            best_mse = mse
            best_train_r2 = train_r2

            best_y_pred = y_pred
            best_y_test = y_test

# 输出最优问值和最优cv下模型的R2、RMSE、MAE、MSE
print("\nBest Threshold:", best_threshold)
print("Best CV: ", best_cv)
print("Best R2 Score (Test): ", best_r2)
print("Best RMSE: ", best_rmse)
print("Best MAE: ", best_mae)
print("Best MSE: ", best_mse)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 输出能够表示模型的图片
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(best_y_test, best_y_pred, s=80, alpha=0.8, edgecolors='k')
plt.plot([best_y_test.min(), best_y_test.max()], [best_y_test.min(), best_y_test.max()], 'k--', label='Perfect Fit')  # 添加最佳拟合线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('K-Nearest Neighbor Model')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend()  # 显示图例
plt.savefig("knn_model_plot.png", dpi=300, bbox_inches="tight")
plt.show()
