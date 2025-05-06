import numpy as np
import matplotlib.pyplot as plt
# 解决 Matplotlib 中文显示问题（如果需要）
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 1. 定义 Sigmoid 函数
def sigmoid(z):
    """
    计算 sigmoid 函数值
    Args:
        z (ndarray): 线性组合 (w*x + b) 的结果，可以是标量或数组
    Returns:
        ndarray: Sigmoid 函数的输出，范围在 0 到 1 之间
    """
    return 1 / (1 + np.exp(-z))

# 2. 定义成本函数 (Log Loss / Binary Cross-Entropy)
def compute_cost(X, y, w, b):
    """
    计算逻辑回归的成本函数
    Args:
        X (ndarray): 特征数据，形状 (m, n)，m 是样本数，n 是特征数
        y (ndarray): 实际标签 (0 或 1)，形状 (m, 1)
        w (ndarray): 权重，形状 (n, 1)
        b (float):   偏差 (截距)
    Returns:
        float: 当前参数下的成本值
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)

    # 加上一个极小值 epsilon 防止 log(0) 错误
    epsilon = 1e-5
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# 3. 定义梯度下降函数
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    """
    使用梯度下降法更新权重 w 和偏差 b
    Args:
        X (ndarray): 特征数据，形状 (m, n)
        y (ndarray): 实际标签，形状 (m, 1)
        w (ndarray): 初始权重，形状 (n, 1)
        b (float):   初始偏差
        learning_rate (float): 学习率
        num_iterations (int):  迭代次数
    Returns:
        tuple: 包含最终权重、最终偏差和成本历史记录的元组
    """
    m = X.shape[0]
    costs = [] # 用来记录每次迭代的成本

    for i in range(num_iterations):
        # 计算预测值 h 和线性组合 z
        z = np.dot(X, w) + b
        h = sigmoid(z)

        # 计算梯度
        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每隔一定次数记录成本
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            costs.append(cost)
            # print(f"Iteration {i}: Cost = {cost}")

    return w, b, costs

# 4. 定义预测函数
def predict(X, w, b, threshold=0.5):
    """
    根据学习到的参数进行预测
    Args:
        X (ndarray): 要预测的数据，形状 (m, n)
        w (ndarray): 学习到的权重，形状 (n, 1)
        b (float):   学习到的偏差
        threshold (float): 判断类别的阈值，默认为 0.5
    Returns:
        ndarray: 预测的类别 (0 或 1)，形状 (m, 1)
    """
    m = X.shape[0]
    predictions = np.zeros((m, 1))

    z = np.dot(X, w) + b
    h = sigmoid(z)

    # 将概率大于等于 threshold 的预测为 1，否则为 0
    predictions[h >= threshold] = 1

    return predictions

# 5. 生成模拟数据 (二维特征)
np.random.seed(1) # 为了可重现性
m = 150 # 样本数
n = 2   # 特征数

# 产生两群数据点
X = np.random.randn(m, n) * 1.5
y = np.zeros((m, 1))

# 创建一个大致的线性边界，例如 x1 + x2 > 1
boundary_condition = X[:, 0] + X[:, 1] > 1
y[boundary_condition] = 1

# 为数据增加一些偏移和噪音，让分界不完美
X[y[:, 0] == 0, :] -= 1 # 将类别 0 的点向左下方移动
X[y[:, 0] == 1, :] += 1 # 将类别 1 的点向右上方移动
X += np.random.randn(m, n) * 0.3 # 加入一些噪音

# 6. 初始化参数
w_init = np.zeros((n, 1))
b_init = 0.0

# 7. 设定超参数
learning_rate = 0.01
num_iterations = 10000

# 8. 执行梯度下降训练模型
w_final, b_final, costs = gradient_descent(X, y, w_init, b_init, learning_rate, num_iterations)

print(f"训练完成后的权重 w: {w_final.flatten()}")
print(f"训练完成后的偏差 b: {b_final}")
print(f"最终成本: {costs[-1]}")

# 9. 绘制结果
plt.figure(figsize=(10, 6))

# 绘制数据点
# 将 y 转换为 boolean 索引
y_bool = y.astype(bool).flatten()
plt.scatter(X[y_bool, 0], X[y_bool, 1], c='blue', label='类别 1', marker='o')
plt.scatter(X[~y_bool, 0], X[~y_bool, 1], c='red', label='类别 0', marker='x')

# 绘制决策边界
# 决策边界是 w1*x1 + w2*x2 + b = 0
# 我们需要绘制 x2 = (-w1*x1 - b) / w2
x1_plot = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100) # 生成 x1 的范围

# 检查 w_final[1] 是否接近 0，以避免除以零
if np.abs(w_final[1, 0]) > 1e-6:
    x2_plot = (-w_final[0, 0] * x1_plot - b_final) / w_final[1, 0]
    plt.plot(x1_plot, x2_plot, label='决策边界', color='green', linewidth=2)
elif np.abs(w_final[0, 0]) > 1e-6:
    # 如果 w2 接近 0，则决策边界是垂直线 x1 = -b / w1
    x1_boundary = -b_final / w_final[0, 0]
    plt.axvline(x=x1_boundary, label='决策边界 (垂直)', color='green', linewidth=2)
else:
    print("警告：权重 w1 和 w2 都接近于 0，无法绘制标准决策边界。")


# 设定图表属性
plt.xlabel('特征 1 (X1)')
plt.ylabel('特征 2 (X2)')
plt.title('NumPy 实现的逻辑回归二分类与决策边界')
plt.legend()
plt.grid(True)
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.show()

# (可选) 绘制成本函数随迭代次数的变化
plt.figure(figsize=(8, 5))
plt.plot(range(0, num_iterations, 100), costs) # 注意 x 轴对应的是记录成本的迭代次数
plt.xlabel('迭代次数 (每 100 次)')
plt.ylabel('成本函数值')
plt.title('成本函数随训练迭代的变化')
plt.grid(True)
plt.show()