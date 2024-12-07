import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 定义一个三次函数模型
def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 读取Excel文件的路径保持不变
file_path = r"C:\Users\27144\Desktop\中-tmin.xlsx"
data = pd.read_excel(file_path)

# 将'F-M TMIN'列四舍五入到整数
data['F-M TMIN'] = np.round(data['F-M TMIN'] * 2) / 2
data['F-M TMIN'] = data['F-M TMIN'].astype(int)

# 对于每个'F-M TMIN'值，计算'Chalkiness'的平均值和标准差
grouped_data = data.groupby('F-M TMIN')['Chalkiness'].agg(['mean', 'std']).reset_index()

x_unique = grouped_data['F-M TMIN']
y_mean = grouped_data['mean']
y_std = grouped_data['std']

# 使用curve_fit进行非线性拟合，这次使用三次函数
params, _ = curve_fit(cubic_func, x_unique, y_mean)
a, b, c, d = params
print(f"Cubic Curve parameters: a = {a}, b = {b}, c = {c}, d = {d}")


plt.rcParams['figure.dpi'] = 600

# 计算拟合曲线上的y值
y_fit = cubic_func(x_unique, a, b, c, d)

# 计算R²分数以评估模型与数据的拟合度
r2 = r2_score(y_mean, y_fit)
print(f"R² Score: {r2}")

# 绘制拟合曲线，不包含原始散点，但会添加误差棒
plt.figure(figsize=(10, 6))
plt.errorbar(x_unique, y_mean, yerr=y_std, fmt='o', capsize=5, label='Chalkiness', color='blue', alpha=0.6)

plt.plot(x_unique, y_fit, 'r-', label=f'Fit: y = {a:.4f}x^3 + {b:.2f}x^2 + {c:.2f}x + {d:.2f}', linewidth=2)

# 添加R²分数到图中
r2_text_position = (0.3, 0.8)  # 位置参数，(x, y)，其中x=0.98表示接近右侧边缘，y=0.05表示底部边缘以上
plt.text(r2_text_position[0], r2_text_position[1], f'R² = {r2:.4f}', transform=plt.gca().transAxes,
         horizontalalignment='right', verticalalignment='bottom', fontsize=16)




# 添加图表标题和坐标轴标签

plt.xlabel('F-M TMIN')
plt.ylabel('Chalkiness')

# 显示图例
plt.legend()

# 优化x轴刻度避免重叠
plt.xticks(rotation=45)

# 显示图形
plt.tight_layout()
plt.show()