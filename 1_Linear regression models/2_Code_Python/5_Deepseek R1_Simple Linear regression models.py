import numpy as np
import matplotlib.pyplot as plt
# === 1. 生成模拟数据 ===
np.random.seed(42)  # 固定随机种子，确保结果可复现
x = np.linspace(0, 10, 100)          # 生成100个x值（0到10之间）
y_true = 2 * x + 3                   # 真实关系：y = 2x + 3
y_noise = np.random.normal(0, 2, 100) # 添加高斯噪声（均值0，标准差2）
y = y_true + y_noise                 # 观测到的含噪声的y值
# === 2. 构造设计矩阵 X（带截距项） ===
X = np.c_[np.ones(x.shape), x]  # X的第1列全为1（对应截距项 θ0），第2列是x
# === 3. 使用正规方程求解最优参数 θ ===
theta = np.linalg.inv(X.T @ X) @ X.T @ y  # θ = (X^T X)^{-1} X^T y
# === 4. 预测拟合值 ===
y_pred = X @ theta  # 等价于 theta[0] + theta[1] * x
# === 5. 可视化结果 ===
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Noisy Data')      # 绘制原始数据点
plt.plot(x, y_true, 'g--', label='True Line (y=2x+3)')   # 真实关系线
plt.plot(x, y_pred, 'r-', label='Fitted Line')           # 拟合的线性回归线
plt.title('Linear Regression with Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
# === 输出结果 ===
print("最优参数：θ0 =", theta[0], "θ1 =", theta[1])
