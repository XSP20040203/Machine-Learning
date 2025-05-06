import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# === 1. 生成模拟数据 ===
np.random.seed(42)
n_samples = 100  # 样本量
# 生成两个特征 x1, x2
x1 = np.random.uniform(0, 10, n_samples)
x2 = np.random.uniform(5, 15, n_samples)
noise = np.random.normal(0, 3, n_samples)  # 噪声
# 真实关系：y = 3 + 2x1 + 1.5x2
y_true = 3 + 2 * x1 + 1.5 * x2
y = y_true + noise  # 观测到的 y（含噪声）
# === 2. 构造设计矩阵 X（包含截距项、x1、x2）===
X = np.column_stack([np.ones(n_samples), x1, x2])
# === 3. 正规方程法求解参数 θ ===
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"最优参数：θ0 = {theta[0]:.2f}, θ1 = {theta[1]:.2f}, θ2 = {theta[2]:.2f}")
# ==== 4. 可视化结果 ====
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# 绘制原始数据点（蓝色）
ax.scatter(x1, x2, y, c='blue', label='Observed Data')
# 生成网格点用于绘制平面
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 10, 20), np.linspace(5, 15, 20))
y_pred_grid = theta[0] + theta[1]*x1_grid + theta[2]*x2_grid
# 绘制拟合平面（红色半透明）
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='r', alpha=0.5, label='Fitted Plane')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Multivariate Linear Regression (2 features)')
ax.legend()
plt.show()
