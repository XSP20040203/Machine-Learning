import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成简单数据
x = np.linspace(0, 10, 100)  # 生成 100 个 x 值，从 0 到 10
true_beta0 = 2               # 真实的截距
true_beta1 = 3               # 真实的斜率
noise = np.random.normal(0, 1, 100)  # 添加正态分布噪声
y = true_beta0 + true_beta1 * x + noise  # 生成 y 数据

# 计算 x 和 y 的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 用最小二乘法估计参数
numerator = np.sum((x - x_mean) * (y - y_mean))  # 分子
denominator = np.sum((x - x_mean) ** 2)          # 分母
beta1_hat = numerator / denominator              # 估计斜率
beta0_hat = y_mean - beta1_hat * x_mean          # 估计截距

print(f"估计的截距 (beta0): {beta0_hat:.2f}")
print(f"估计的斜率 (beta1): {beta1_hat:.2f}")

# 用估计参数预测 y
y_pred = beta0_hat + beta1_hat * x

# 绘制图形
plt.scatter(x, y, label="数据点", color="blue", alpha=0.5)  # 原始数据散点
plt.plot(x, true_beta0 + true_beta1 * x, color="green", label="真实直线")  # 真实关系
plt.plot(x, y_pred, color="red", label="估计回归线")  # 估计回归线
plt.xlabel("x")
plt.ylabel("y")
plt.title("线性回归示例")
plt.legend()
plt.show()

# 计算模型拟合优度 R^2
residuals = y - y_pred              # 残差
tss = np.sum((y - y_mean) ** 2)    # 总平方和
rss = np.sum(residuals ** 2)        # 残差平方和
r_squared = 1 - (rss / tss)        # 决定系数
print(f"R^2: {r_squared:.4f}")