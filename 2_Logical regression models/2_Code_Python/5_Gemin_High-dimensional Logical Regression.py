import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA # 导入 PCA
# 解决 Matplotlib 中文显示问题（如果需要，请取消注释并确保安装了相应字体库，如 SimHei）
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- Softmax 回归核心函数定义 ---

# 辅助函数：One-Hot 编码
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

# 定义 Softmax 函数 (考虑数值稳定性)
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

# 定义成本函数 (Categorical Cross-Entropy)
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

# 定义梯度下降函数
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
            # 可选：打印迭代过程中的成本
            # if i % 500 == 0:
            #    print(f"Iteration {i}: Cost = {cost}")

    return W, b, costs

# 定义预测函数
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

# --- --- --- --- --- --- --- --- --- --- ---

# 1. 数据准备：生成具有多个特征的模拟数据
N_SAMPLES = 400
N_FEATURES_ORIGINAL = 4 # *** 使用 4 个特征 ***
N_CLASSES = 3
RANDOM_SEED = 42

X, y_int = make_blobs(n_samples=N_SAMPLES,
                       n_features=N_FEATURES_ORIGINAL, # 使用 4 个特征
                       centers=N_CLASSES,
                       cluster_std=1.5, # 簇的标准差，影响分散程度
                       random_state=RANDOM_SEED) # 保证结果可复现

print(f"原始数据 X 的形状: {X.shape}")
print(f"标签 y_int 的形状: {y_int.shape}")
print(f"类别数量: {len(np.unique(y_int))}")

# 转换为 One-Hot 编码
y_one_hot = to_one_hot(y_int, N_CLASSES)
print(f"One-Hot 编码后 y_one_hot 的形状: {y_one_hot.shape}")

# 2. 在原始高维数据 (4维) 上训练 Softmax 模型
print("\n--- 在原始 4D 数据上训练模型 ---")
# 初始化参数
n_features = X.shape[1]
n_classes = y_one_hot.shape[1]
np.random.seed(0)
W_init = np.random.randn(n_features, n_classes) * 0.01
b_init = np.zeros((1, n_classes))

# 设定超参数
learning_rate_high_dim = 0.3 # 学习率可能需要调整
num_iterations_high_dim = 15000 # 迭代次数可能需要调整

# 执行梯度下降训练模型
W_final, b_final, costs_high_dim = gradient_descent(
    X, y_one_hot, W_init, b_init,
    learning_rate_high_dim, num_iterations_high_dim
)

print(f"高维模型训练完成后的最终成本: {costs_high_dim[-1]}")

# 获取高维模型在原始数据上的预测结果
y_pred_high_dim = predict(X, W_final, b_final)
accuracy_high_dim = np.mean(y_pred_high_dim == y_int) * 100
print(f"高维模型在训练集上的准确率: {accuracy_high_dim:.2f}%")


# 3. 使用 PCA 将数据降至 2 维
print("\n--- 使用 PCA 进行降维 ---")
pca = PCA(n_components=2) # 创建 PCA 对象，指定降维到 2 个主成分
X_pca = pca.fit_transform(X) # 对原始数据 X 进行拟合和转换

print(f"PCA 降维后数据 X_pca 的形状: {X_pca.shape}")


# 4. 可视化 PCA 降维结果
print("\n--- 可视化 PCA 降维结果 ---")
plt.figure(figsize=(14, 6))

# 图 1: PCA 结果，按真实类别着色
plt.subplot(1, 2, 1)
scatter_true = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_int, cmap=plt.cm.Spectral, edgecolor='k', s=40)
plt.xlabel('主成分 1 (Principal Component 1)')
plt.ylabel('主成分 2 (Principal Component 2)')
plt.title('PCA 降维结果 (按真实类别着色)')
plt.grid(alpha=0.3)
# 创建图例
if N_CLASSES <= 10:
   handles_true, labels_true = scatter_true.legend_elements(prop="colors")
   legend_labels_true = [f'类别 {i}' for i in range(N_CLASSES)]
   plt.legend(handles_true, legend_labels_true, title="真实类别")

# 图 2: PCA 结果，按高维模型预测类别着色
# 注意：这里用的是在高维数据上训练的模型的预测结果 y_pred_high_dim
plt.subplot(1, 2, 2)
scatter_pred = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_high_dim, cmap=plt.cm.Spectral, edgecolor='k', s=40)
plt.xlabel('主成分 1 (Principal Component 1)')
plt.ylabel('主成分 2 (Principal Component 2)')
plt.title('PCA 降维结果 (按高维模型预测着色)')
plt.grid(alpha=0.3)
# 创建图例
if N_CLASSES <= 10:
   handles_pred, labels_pred = scatter_pred.legend_elements(prop="colors")
   legend_labels_pred = [f'类别 {i}' for i in range(N_CLASSES)]
   plt.legend(handles_pred, legend_labels_pred, title="预测类别")

plt.tight_layout() # 调整子图布局
plt.show()


# 5. (说明性) 在 2D PCA 数据上训练新模型并绘制决策边界
print("\n--- 在 2D PCA 数据上训练新模型以绘制说明性决策边界 ---")

# 初始化用于 2D PCA 数据的新参数
n_features_pca = X_pca.shape[1] # 应该是 2
np.random.seed(1) # 使用不同种子以区别于高维模型
W_init_pca = np.random.randn(n_features_pca, N_CLASSES) * 0.01
b_init_pca = np.zeros((1, N_CLASSES))

# 使用 PCA 数据训练新模型 (注意：输入是 X_pca)
learning_rate_pca = 0.5 # PCA 后的数据可能适用不同的学习率
num_iterations_pca = 1000 # 迭代次数也可能需要调整

W_final_pca, b_final_pca, costs_pca = gradient_descent(
    X_pca, y_one_hot, W_init_pca, b_init_pca,
    learning_rate_pca, num_iterations_pca
)

print(f"在 2D PCA 数据上训练的模型的最终成本: {costs_pca[-1]}")
y_pred_pca = predict(X_pca, W_final_pca, b_final_pca)
accuracy_pca = np.mean(y_pred_pca == y_int) * 100
print(f"在 2D PCA 数据上训练的模型的准确率: {accuracy_pca:.2f}%")

# 定义绘制决策区域的函数（与之前类似，但输入是 PCA 数据和 PCA 模型参数）
def plot_decision_regions_pca(X_pca, y_int, W, b, title):
    """绘制 PCA 数据的决策区域"""
    plt.figure(figsize=(10, 7))
    # 设置绘制范围
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    # 生成网格点
    h = 0.02 # 网格步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

    # 对网格中每个点进行预测
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # 使用传入的 W, b (在 PCA 数据上训练得到的) 进行预测
    Z = predict(grid_points, W, b)
    Z = Z.reshape(xx.shape) # 将预测结果 reshape 成网格形状

    # 绘制决策区域的填充轮廓
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)

    # 绘制原始数据点（使用 PCA 坐标），按真实类别着色
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_int, cmap=plt.cm.Spectral,
                          edgecolor='k', s=40) # s 是点的大小

    plt.xlabel('主成分 1 (PC1)')
    plt.ylabel('主成分 2 (PC2)')
    plt.title(title)

    # 创建图例（如果类别不多）
    n_classes_plot = len(np.unique(y_int))
    if n_classes_plot <= 10: # 类别太多图例会很乱
       handles, labels = scatter.legend_elements(prop="colors")
       legend_labels = [f'类别 {i}' for i in range(n_classes_plot)]
       plt.legend(handles, legend_labels, title="真实类别")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid(alpha=0.3)
    plt.show()

# 调用绘图函数，显示在 2D PCA 数据上训练的模型的决策边界
plot_decision_regions_pca(
    X_pca, y_int, W_final_pca, b_final_pca,
    title='在 2D PCA 数据上训练的 Softmax 决策边界 (说明性)'
)

# (可选) 绘制高维模型训练过程中的成本函数变化
plt.figure(figsize=(8, 5))
plt.plot(range(0, num_iterations_high_dim, 100), costs_high_dim) # 每 100 次迭代记录一次成本
plt.xlabel('迭代次数 (每 100 次)')
plt.ylabel('成本函数值 (交叉熵)')
plt.title('高维模型训练过程中的成本变化')
plt.grid(True)
plt.show()