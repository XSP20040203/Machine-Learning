import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
'''
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
'''
# 示例数据集
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
n = len(X)

print(X)
print(y)

# -------------------- 梯度下降实现 (Gradient Descent) --------------------
theta_0_gd = 0
theta_1_gd = 0
learning_rate = 0.01
iterations = 100

for i in range(iterations):
    y_predicted = theta_0_gd + theta_1_gd * X # 预测值

    d_theta_0 = (1/n) * np.sum(y_predicted - y) # 计算梯度

    d_theta_1 = (1/n) * np.sum((y_predicted - y) * X) # 计算梯度

    theta_0_gd = theta_0_gd - learning_rate * d_theta_0 # 更新参数

    theta_1_gd = theta_1_gd - learning_rate * d_theta_1 # 更新参数

print("\n梯度下降结果:")
print(f"theta_0 (截距): {theta_0_gd:.4f}")
print(f"theta_1 (斜率): {theta_1_gd:.4f}")

# -------------------- 正规方程实现 (Normal Equation) --------------------
X_b = np.c_[np.ones((n, 1)), X]
y_col = y.reshape(-1, 1)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_col)
theta_0_ne = theta_best[0][0]
theta_1_ne = theta_best[1][0]

print("\n正规方程结果:")
print(f"theta_0 (截距): {theta_0_ne:.4f}")
print(f"theta_1 (斜率): {theta_1_ne:.4f}")

# -------------------- 图像化呈现 --------------------
plt.figure(figsize=(10, 6)) # 设置图像大小

# 绘制原始数据点
plt.scatter(X, y, color='blue', label='Original Data')

# 绘制梯度下降的回归线
y_pred_gd = theta_0_gd + theta_1_gd * X
plt.plot(X, y_pred_gd, color='red', label=f'Gradient Descent ($\\theta_0$={theta_0_gd:.2f}, $\\theta_1$={theta_1_gd:.2f})')

# 绘制正规方程的回归线
y_pred_ne = theta_0_ne + theta_1_ne * X 
plt.plot(X, y_pred_ne, color='green', linestyle='--', label=f'Normal Equation ($\\theta_0$={theta_0_ne:.2f}, $\\theta_1$={theta_1_ne:.2f})')

# 添加标签和标题
plt.xlabel('X') # 设置x轴标签
plt.ylabel('y') # 设置y轴标签
plt.title('Linear Regression with Gradient Descent and Normal Equation')
plt.legend() # 添加图例
plt.grid(True) # 添加网格线
plt.show() # 显示图像

# -------------------- 模型评估 --------------------

# 计算梯度下降模型的指标
mse_gd = mean_squared_error(y, y_pred_gd)
mae_gd = mean_absolute_error(y, y_pred_gd)
r2_gd = r2_score(y, y_pred_gd)

# 计算正规方程模型的指标
mse_ne = mean_squared_error(y, y_pred_ne)
mae_ne = mean_absolute_error(y, y_pred_ne)
r2_ne = r2_score(y, y_pred_ne)

print("\n梯度下降模型评估:")
print(f"MSE: {mse_gd:.4f}, MAE: {mae_gd:.4f}, R²: {r2_gd:.4f}")

print("\n正规方程模型评估:")
print(f"MSE: {mse_ne:.4f}, MAE: {mae_ne:.4f}, R²: {r2_ne:.4f}")

# -------------------- 评估指标可视化 --------------------
metrics = ['MSE', 'MAE', 'R²']
gd_scores = [mse_gd, mae_gd, r2_gd]
ne_scores = [mse_ne, mae_ne, r2_ne]

x = np.arange(len(metrics))  # 指标标签位置
width = 0.35  # 柱状图宽度

plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, gd_scores, width, label='gradient descent', color='#FF6B6B')
rects2 = plt.bar(x + width/2, ne_scores, width, label='normal equation', color='#4ECDC4')

# 添加标签和美化
plt.title('Comparison of model performance', fontsize=14, pad=20)
plt.xlabel('Evaluation Index', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(x, metrics)
plt.ylim(0, max(max(gd_scores), max(ne_scores)) * 1.2)

# 在柱子上方显示数值
for rect in rects1 + rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.legend(frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()