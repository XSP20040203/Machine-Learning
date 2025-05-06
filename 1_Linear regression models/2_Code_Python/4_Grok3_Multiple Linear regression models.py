import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成简单数据
n = 100  # 样本数
x1 = np.random.uniform(0, 10, n)  # 特征 x1
x2 = np.random.uniform(0, 10, n)  # 特征 x2
true_beta0 = 2                    # 真实的截距
true_beta1 = 3                    # 真实的 x1 系数
true_beta2 = -1                   # 真实的 x2 系数
noise = np.random.normal(0, 1, n) # 添加正态分布噪声
y = true_beta0 + true_beta1 * x1 + true_beta2 * x2 + noise  # 生成目标变量 y

# 构建特征矩阵 X，包含截距项
X = np.column_stack((np.ones(n), x1, x2))  # 大小为 (n, 3)

# 用最小二乘法估计参数
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
beta0_hat, beta1_hat, beta2_hat = beta_hat

# 输出估计的参数
print(f"估计的截距 (beta0): {beta0_hat:.2f}")
print(f"估计的 x1 系数 (beta1): {beta1_hat:.2f}")
print(f"估计的 x2 系数 (beta2): {beta2_hat:.2f}")

# 用估计参数预测 y
y_pred = X @ beta_hat

# 绘制三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(x1, x2, y, color='blue', label='数据点')

# 绘制回归平面
x1_range = np.linspace(0, 10, 10)
x2_range = np.linspace(0, 10, 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y_grid = beta0_hat + beta1_hat * x1_grid + beta2_hat * x2_grid
ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.5, label='回归平面')

# 设置坐标轴标签和标题
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('多元线性回归示例')

# 显示图形
plt.show()

# 计算模型拟合优度 R^2
y_mean = np.mean(y)
tss = np.sum((y - y_mean) ** 2)  # 总平方和
rss = np.sum((y - y_pred) ** 2)   # 残差平方和
r_squared = 1 - (rss / tss)
print(f"R^2: {r_squared:.4f}")