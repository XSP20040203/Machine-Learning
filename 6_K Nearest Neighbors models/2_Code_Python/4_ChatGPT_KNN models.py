import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ========================
# 1. 生成多分类合成数据集
# ========================
np.random.seed(42)
n_samples = 100
# 定义三个类别的中心
centers = [(-2, 0), (2, 0), (0, 2)]
# 生成每个类别的二维数据
X_list = [np.random.randn(n_samples, 2) + np.array(center) for center in centers]
# 合并数据和标签\ nX = np.vstack(X_list)
y = np.hstack([[i] * n_samples for i in range(len(centers))])

# 数据集可视化
plt.figure(figsize=(6, 6))
colors = ['blue', 'red', 'green']
cmap_bold = ListedColormap(colors)
for idx, Xi in enumerate(X_list):
    plt.scatter(Xi[:, 0], Xi[:, 1], c=colors[idx], edgecolor='k', label=f'Class {idx}')
plt.title('Dataset Visualization (3 Classes)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# ========================
# 2. 实现 KNN 算法（支持多分类）
# ========================
def knn_predict(X_train, y_train, X_test, k=5, p=2):
    """
    使用 Minkowski 距离实现 KNN 多分类
    X_train: shape=(n_train, d)
    y_train: shape=(n_train,)
    X_test: shape=(n_test, d)
    返回: y_pred shape=(n_test,)
    """
    n_test = X_test.shape[0]
    y_pred = np.empty(n_test, dtype=int)
   
    for i in range(n_test):
        dist = np.linalg.norm(X_train - X_test[i], ord=p, axis=1)
        nn_idx = np.argsort(dist)[:k]
        votes = y_train[nn_idx]
        labels, counts = np.unique(votes, return_counts=True)
        y_pred[i] = labels[np.argmax(counts)]
    return y_pred

# ========================
# 3. 模型决策边界可视化
# ========================

# 构建网格
# ========================
# 3. 模型决策边界可视化
# ========================

# 修正合并数据集 (原代码第11行存在注释错误)
X = np.vstack(X_list)  # 取消原第11行的错误注释

# 构建网格
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 修正nx_min拼写错误
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 网格预测
Z = knn_predict(X, y, grid_points, k=5)
Z = Z.reshape(xx.shape)

# 可视化决策区域
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
# 绘制训练点
for idx, Xi in enumerate(X_list):
    plt.scatter(Xi[:, 0], Xi[:, 1], c=colors[idx], edgecolor='k', label=f'Class {idx}')
plt.title('KNN Decision Boundary (k=5)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# ========================
# 4. 多分类测试点预测及可视化
# ========================
# 随机生成测试点
X_test = np.random.randn(30, 2) * 1.5
y_test_pred = knn_predict(X, y, X_test, k=5)

plt.figure(figsize=(6, 6))
# 绘制训练点
for idx, Xi in enumerate(X_list):
    plt.scatter(Xi[:, 0], Xi[:, 1], c=colors[idx], edgecolor='k', label=f'Class {idx} Train')
# 绘制测试点，按预测类别上色
for cls in np.unique(y_test_pred):
    pts = X_test[y_test_pred == cls]
    plt.scatter(pts[:, 0], pts[:, 1], c=colors[cls], marker='x', s=100,
                label=f'Predicted {cls}')
plt.title('Test Points Classification (Multi-Class)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
