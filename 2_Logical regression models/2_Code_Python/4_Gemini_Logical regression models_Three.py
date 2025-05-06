import numpy as np
import matplotlib.pyplot as plt
# 解决 Matplotlib 中文显示问题（如果需要）
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 1. 数据准备：生成模拟的多类数据
# 为了方便生成清晰的多类数据，我们这里暂时借用 sklearn 的 make_blobs
# 但核心的 Softmax 回归算法完全使用 NumPy 实现
from sklearn.datasets import make_blobs

N_SAMPLES = 1000 # 样本总数
N_FEATURES = 2  # 特征数
N_CLASSES = 5   # 类别数

X, y_int = make_blobs(n_samples=N_SAMPLES,
                       n_features=N_FEATURES,
                       centers=N_CLASSES,
                       cluster_std=1.2, # 让簇稍微分散一点
                       random_state=42) # 保证结果可复现

# 2. 辅助函数：One-Hot 编码
def to_one_hot(y_int, n_classes):
    """
    将整数标签转换为 One-Hot 编码
    Args:
        y_int (ndarray): 包含整数标签 (0 to n_classes-1) 的数组，形状 (m,)
        n_classes (int): 类别总数
    Returns:
        ndarray: One-Hot 编码后的矩阵，形状 (m, n_classes)
    """
    m = y_int.shape[0]
    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y_int] = 1
    return y_one_hot

# 将 y_int 转换为 One-Hot 编码
y_one_hot = to_one_hot(y_int, N_CLASSES)
# y_int 保留整数形式，用于绘图时的颜色区分和预测结果对比

# 3. 定义 Softmax 函数 (考虑数值稳定性)
def softmax(z):
    """
    计算 Softmax 函数值
    Args:
        z (ndarray): 线性得分，形状 (m, n_classes)
    Returns:
        ndarray: 各类别的概率，形状 (m, n_classes)，每行加总为 1
    """
    # 减去最大值防止 exp() 溢出
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 4. 定义成本函数 (Categorical Cross-Entropy)
def compute_cost(X, y_one_hot, W, b):
    """
    计算 Softmax 回归的交叉熵成本
    Args:
        X (ndarray): 特征数据，形状 (m, n)
        y_one_hot (ndarray): 真实标签的 One-Hot 编码，形状 (m, k) k=类别数
        W (ndarray): 权重矩阵，形状 (n, k)
        b (ndarray): 偏差向量，形状 (1, k)
    Returns:
        float: 成本值
    """
    m = X.shape[0]
    # 计算线性得分
    scores = X @ W + b # 形状 (m, k)
    # 计算概率
    probabilities = softmax(scores) # 形状 (m, k)

    # 计算交叉熵成本（加上 epsilon 防止 log(0)）
    epsilon = 1e-9
    cost = (-1/m) * np.sum(y_one_hot * np.log(probabilities + epsilon))
    return cost

# 5. 定义梯度下降函数
def gradient_descent(X, y_one_hot, W, b, learning_rate, num_iterations):
    """
    使用梯度下降法更新 Softmax 回归的参数 W 和 b
    Args:
        X (ndarray): 特征数据，形状 (m, n)
        y_one_hot (ndarray): 真实标签的 One-Hot 编码，形状 (m, k)
        W (ndarray): 初始权重矩阵，形状 (n, k)
        b (ndarray): 初始偏差向量，形状 (1, k)
        learning_rate (float): 学习率
        num_iterations (int): 迭代次数
    Returns:
        tuple: 包含最终权重、最终偏差和成本历史记录的元组
    """
    m = X.shape[0]
    costs = []

    for i in range(num_iterations):
        # 计算线性得分和概率
        scores = X @ W + b
        probabilities = softmax(scores)

        # 计算梯度
        # dScores 是成本函数对 scores 的梯度
        dScores = probabilities - y_one_hot # 形状 (m, k)
        dW = (1/m) * (X.T @ dScores)       # 形状 (n, k)
        db = (1/m) * np.sum(dScores, axis=0, keepdims=True) # 形状 (1, k)

        # 更新参数
        W = W - learning_rate * dW
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            cost = compute_cost(X, y_one_hot, W, b)
            costs.append(cost)
            # print(f"Iteration {i}: Cost = {cost}")

    return W, b, costs

# 6. 定义预测函数
def predict(X, W, b):
    """
    根据学习到的参数进行多分类预测
    Args:
        X (ndarray): 要预测的数据，形状 (m, n)
        W (ndarray): 学习到的权重矩阵，形状 (n, k)
        b (ndarray): 学习到的偏差向量，形状 (1, k)
    Returns:
        ndarray: 预测的类别整数标签 (0 to k-1)，形状 (m,)
    """
    scores = X @ W + b
    probabilities = softmax(scores)
    # 返回概率最高的类别的索引
    return np.argmax(probabilities, axis=1)

# 7. 初始化参数
n_features = X.shape[1]
n_classes = y_one_hot.shape[1]

# 使用小的随机数初始化，有助于打破对称性
np.random.seed(0)
W_init = np.random.randn(n_features, n_classes) * 0.01
b_init = np.zeros((1, n_classes))

# 8. 设定超参数
learning_rate = 0.01 # Softmax 可能需要稍大的学习率
num_iterations = 20000

# 9. 执行梯度下降训练模型
W_final, b_final, costs = gradient_descent(X, y_one_hot, W_init, b_init, learning_rate, num_iterations)

print(f"训练完成后的最终成本: {costs[-1]}")

# 10. 评估模型 (可选)
y_pred = predict(X, W_final, b_final)
accuracy = np.mean(y_pred == y_int) * 100
print(f"训练集上的准确率: {accuracy:.2f}%")

# 11. 绘制决策区域和数据点
def plot_decision_regions(X, y_int, W, b):
    """绘制数据点和 Softmax 回归的决策区域"""
    plt.figure(figsize=(10, 7))

    # 设置绘制范围，稍微超出数据点范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成网格点
    h = 0.02 # 网格步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

    # 对网格中每个点进行预测
    # np.c_ 按列合并，ravel() 将网格展平
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid_points, W, b)
    # 将预测结果 reshape 成网格形状
    Z = Z.reshape(xx.shape)

    # 绘制决策区域的填充轮廓
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)

    # 绘制原始数据点，按真实类别着色
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_int, cmap=plt.cm.Spectral,
                          edgecolor='k', s=40) # s 是点的大小

    plt.xlabel('特征 1 (X1)')
    plt.ylabel('特征 2 (X2)')
    plt.title('NumPy 实现的 Softmax 回归多分类与决策区域')

    # 创建图例（如果类别不多）
    n_classes = len(np.unique(y_int))
    if n_classes <= 10: # 类别太多图例会很乱
       handles, labels = scatter.legend_elements(prop="colors")
       legend_labels = [f'类别 {i}' for i in range(n_classes)]
       plt.legend(handles, legend_labels, title="真实类别")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid(alpha=0.3)
    plt.show()

# 调用绘图函数
plot_decision_regions(X, y_int, W_final, b_final)

# (可选) 绘制成本函数随迭代次的变化
plt.figure(figsize=(8, 5))
plt.plot(range(0, num_iterations, 100), costs)
plt.xlabel('迭代次数 (每 100 次)')
plt.ylabel('成本函数值 (交叉熵)')
plt.title('成本函数随训练迭代的变化')
plt.grid(True)
plt.show()