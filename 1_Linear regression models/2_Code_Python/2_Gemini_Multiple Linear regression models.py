import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 图形

# -------------------- 生成多元线性回归的示例数据集 --------------------
np.random.seed(0)  # 设置随机种子以确保结果可重复
n_samples = 100
n_features = 2
X = 2 * np.random.rand(n_samples, n_features)
print(X)
true_theta = np.array([1, 2, 3])  # [截距, 特征1系数, 特征2系数]
y = true_theta[0] + X[:, 0] * true_theta[1] + X[:, 1] * true_theta[2] + np.random.randn(n_samples) * 0.5
print(y)
# 添加截距项到 X
X_b = np.c_[np.ones((n_samples, 1)), X]

# -------------------- 多元线性回归的梯度下降实现 --------------------
learning_rate = 0.01
n_iterations = 1000
theta_gd = np.random.randn(n_features + 1, 1)  # 初始化参数

for iteration in range(n_iterations):
    gradients = (1/n_samples) * X_b.T.dot(X_b.dot(theta_gd) - y.reshape(-1, 1))
    theta_gd = theta_gd - learning_rate * gradients

print("\n多元线性回归梯度下降结果:")
print(f"theta_0 (截距): {theta_gd[0][0]:.4f}")
print(f"theta_1 (特征1系数): {theta_gd[1][0]:.4f}")
print(f"theta_2 (特征2系数): {theta_gd[2][0]:.4f}")

# -------------------- 多元线性回归的正规方程实现 --------------------
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y.reshape(-1, 1))

print("\n多元线性回归正规方程结果:")
print(f"theta_0 (截距): {theta_best[0][0]:.4f}")
print(f"theta_1 (特征1系数): {theta_best[1][0]:.4f}")
print(f"theta_2 (特征2系数): {theta_best[2][0]:.4f}")

# -------------------- 图像化呈现 (预测值 vs 实际值) --------------------
y_predict_gd = X_b.dot(theta_gd)
y_predict_ne = X_b.dot(theta_best)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y, y_predict_gd, color='red', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 画一条 y=x 的参考线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values (Gradient Descent)')
plt.title('Predicted vs Actual Values (Gradient Descent)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y, y_predict_ne, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 画一条 y=x 的参考线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values (Normal Equation)')
plt.title('Predicted vs Actual Values (Normal Equation)')
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------- 可选的 3D 可视化 (仅当特征数为 2 时) --------------------
if n_features == 2:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o', label='Original Data')

    # 创建预测平面 (使用正规方程的结果)
    x1_surf = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_surf = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1_surf, x2_surf)
    Y = theta_best[0][0] + theta_best[1][0] * X1 + theta_best[2][0] * X2
    ax.plot_surface(X1, X2, Y, color='green', alpha=0.5, label='Regression Plane (Normal Equation)')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('y')
    ax.set_title('3D Visualization of Multivariate Linear Regression')
    ax.legend()
    plt.show()