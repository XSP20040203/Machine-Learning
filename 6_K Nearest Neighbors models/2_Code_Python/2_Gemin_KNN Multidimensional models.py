import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs # 使用 make_blobs 生成数据
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm # 导入 cm 模块用于获取颜色映射


# 1. 生成模拟数据 (使用 make_blobs)
# 为了方便可视化，这里将 n_features 设置为 2
# 但请注意，下方实现的 KNN 函数是支持多维特征的
n_samples = 300
n_features = 2 # 特征维度设置为 2，便于可视化
n_classes = 4  # 类别数量
X, y = make_blobs(n_samples=n_samples, centers=n_classes, cluster_std=2.0, random_state=42)

# 划分训练集和测试集
test_size = 0.3 # 30% 数据用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print("生成数据总数:", len(X))
print("训练集形状:", X_train.shape, y_train.shape)
print("测试集形状:", X_test.shape, y_test.shape)

# （可选）可视化原始数据 - 如果需要可以取消注释以下代码块
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.8)
# plt.title('Generated Data with True Labels')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# legend1 = plt.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
# plt.gca().add_artist(legend1)
# plt.colorbar(scatter, label='Class ID')
# plt.grid(True)
# plt.show()


# 2. 实现 KNN 分类器功能 (使用函数)
# 计算两个样本点之间的欧几里得距离
def euclidean_distance(x1, x2):
    """
    计算两个样本点之间的欧几里得距离。

    Args:
        x1 (np.ndarray): 第一个样本点。
        x2 (np.ndarray): 第二个样本点。

    Returns:
        float: 欧几里得距离。
    """
    if x1.shape != x2.shape:
        raise ValueError("Points must have the same dimensions for distance calculation.")
    return np.linalg.norm(x1 - x2) # 使用 np.linalg.norm 更简洁高效，且支持多维

def find_k_nearest(x_test_point, X_train, k):
    """
    找到单个测试样本在训练集中的 K 个最近邻居的索引。

    Args:
        x_test_point (np.ndarray): 单个测试样本点。
        X_train (np.ndarray): 训练数据特征矩阵。
        k (int): 近邻的数量。

    Returns:
        np.ndarray: K 个最近邻居在 X_train 中的索引。
    """
    # 确保 k 不超过训练样本数
    k = min(k, X_train.shape[0])
    if k <= 0:
        raise ValueError("k must be a positive integer")


    # 计算当前测试样本与所有训练样本的距离 (向量化计算)
    # X_train 的形状是 (n_train_samples, n_features)
    # x_test_point 的形状是 (n_features,)
    # (X_train - x_test_point) 利用广播机制，形状变为 (n_train_samples, n_features)
    # np.linalg.norm 默认计算 L2 范数（欧几里得距离），axis=1 沿着特征维度计算每个样本的范数
    distances = np.linalg.norm(X_train - x_test_point, axis=1)

    # 获取 K 个最近邻居的索引
    # np.argsort 返回按值升序排序的索引数组
    # [:k] 取出前 K 个索引，对应距离最近的 K 个邻居
    k_indices = np.argsort(distances)[:k]

    return k_indices

def predict_single(x_test_point, X_train, y_train, k):
    """
    预测单个测试样本的类别。

    Args:
        x_test_point (np.ndarray): 单个测试样本点。
        X_train (np.ndarray): 训练数据特征矩阵。
        y_train (np.ndarray): 训练数据标签向量。
        k (int): 近邻的数量。

    Returns:
        int: 预测的类别标签。
    """
    if X_train is None or y_train is None:
        raise RuntimeError("Training data not provided. Ensure X_train and y_train are valid NumPy arrays.")
    if X_train.shape[0] != y_train.shape[0]:
         raise ValueError("Number of samples in X_train and y_train must match.")

    # 找到 K 个最近邻居的索引
    k_indices = find_k_nearest(x_test_point, X_train, k)

    # 获取这 K 个邻居的标签
    k_nearest_labels = y_train[k_indices]

    # 执行多数投票，确定预测类别
    # Counter 统计列表中每个元素的出现次数
    # most_common(1) 返回出现次数最多的元素及其次数组成的列表，如 [(label, count)]
    # [0][0] 提取出列表中第一个元组的第一个元素，即标签
    most_common = Counter(k_nearest_labels).most_common(1)
    predicted_label = most_common[0][0]

    return predicted_label

def predict_knn(X_test, X_train, y_train, k):
    """
    预测多个测试样本的类别。

    Args:
        X_test (np.ndarray): 测试数据特征矩阵。
        X_train (np.ndarray): 训练数据特征矩阵。
        y_train (np.ndarray): 训练数据标签向量。
        k (int): 近邻的数量。

    Returns:
        np.ndarray: 预测的类别标签向量。
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if X_train is None or y_train is None:
        raise RuntimeError("Training data not provided. Ensure X_train and y_train are valid NumPy arrays.")
    if X_train.shape[0] != y_train.shape[0]:
         raise ValueError("Number of samples in X_train and y_train must match.")
    if k > X_train.shape[0]:
         print(f"Warning: k ({k}) is greater than the number of training samples ({X_train.shape[0]}). Prediction will use up to {X_train.shape[0]} neighbors.")


    # 对测试集中的每个样本点调用 predict_single 方法进行预测
    predictions = np.array([predict_single(x, X_train, y_train, k) for x in X_test])
    return predictions


# 3. 进行分类并可视化决策边界和预测值

# 设置 K 值
k_value = 15 # 可以尝试不同的 K 值，它会影响决策边界的平滑度

# 在测试集上进行预测
y_pred = predict_knn(X_test, X_train, y_train, k_value)

# 计算准确率 (可选)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\n测试集准确率 (K={k_value}): {accuracy:.4f}")


# 可视化决策边界和测试数据（按预测标签上色）
plt.figure(figsize=(10, 8))

# 定义网格范围，覆盖所有数据点
# 增加一点边距 (+/- 1)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 生成网格点
h = .1 # 网格步长，越小图越精细但计算量越大
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 对网格中的每个点进行预测
# 使用 ravel() 将网格展平，然后 np.c_ 按列组合成一个二维数组 (n_grid_points, 2)
grid_points = np.c_[xx.ravel(), yy.ravel()]
# 直接调用 predict_knn 函数对所有网格点进行预测
# 注意这里传入的是 X_train 和 y_train，因为是基于训练数据做预测
Z = predict_knn(grid_points, X_train, y_train, k_value)

# 将预测结果 reshape 回网格形状
Z = Z.reshape(xx.shape)

# 获取唯一的类别标签和对应的颜色映射
unique_classes = sorted(np.unique(y))
# 确保颜色映射与类别数匹配
cmap_viridis = plt.get_cmap('viridis', len(unique_classes))
colors = cmap_viridis(np.linspace(0, 1, len(unique_classes)))


# 绘制决策边界的填充区域
# levels 用于指定等高线或填充区域的边界值，这里设置为每个类别中心之间的值
# np.arange(-0.5, len(unique_classes)) 可以创建 [ -0.5, 0.5, 1.5, 2.5, ...] 这样的边界
plt.contourf(xx, yy, Z, levels=np.arange(-0.5, len(unique_classes)), cmap=cmap_viridis, alpha=0.3)


# （可选）绘制训练数据点，作为背景参考
# 使用与决策区域相同的颜色映射，并增加白色边缘
scatter_train = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_viridis,
                            edgecolor='white', s=50, label='Training Data (True)', alpha=0.6)


# 绘制测试数据点，根据预测标签上色，使用圆形标记和黑色边缘
# 使用与决策区域相同的颜色映射
scatter_test_pred = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=cmap_viridis,
                           marker='o', edgecolor='k', s=80, label=f'Test Data (Predicted, K={k_value})', alpha=1.0)


plt.title(f'KNN Decision Boundary with Test Data Predictions (K={k_value})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


# --- 手动创建图例元素 ---
legend_handles = []
# 为每个类别创建图例项（代表决策区域/预测类别）
for i, class_id in enumerate(unique_classes):
    # 创建一个带标记的线条作为图例句柄
    handle = mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None',
                           markersize=10, label=f'Predicted Class {class_id}')
    legend_handles.append(handle)

# 添加训练数据和测试数据的图例项
# 训练数据图例项
handle_train = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                            markeredgecolor='white', markersize=8, label='Training Data (True)')
legend_handles.append(handle_train)

# 测试数据图例项
# 颜色会根据散点图动态决定，这里的颜色只是图例标记的颜色，不代表预测类别
handle_test_pred = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                               markeredgecolor='k', markersize=10, label=f'Test Data (Predicted)')
legend_handles.append(handle_test_pred)


# 将图例添加到图上
plt.legend(handles=legend_handles, loc="lower left", title="Legend")


# 添加颜色条
# 使用测试预测点的散点图对象来创建颜色条，ticks 设置为类别 ID
cbar = plt.colorbar(scatter_test_pred, ticks=unique_classes, label='Predicted Class ID')
cbar.set_ticklabels(unique_classes) # 设置颜色条的标签为类别 ID
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()