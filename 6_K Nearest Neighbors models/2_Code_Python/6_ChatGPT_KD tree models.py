import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# ============================
# 1. 生成多特征合成数据集
# ============================
np.random.seed(0)
n_samples = 200
n_features = 5    # 多维特征
n_classes = 3

# 为每个类别生成中心点
centers = np.random.uniform(-5, 5, size=(n_classes, n_features))

# 生成样本并拆分训练/测试集
X = []
y = []
for idx, center in enumerate(centers):
    Xi = np.random.randn(n_samples, n_features) * 0.8 + center
    X.append(Xi)
    y.append(np.full(n_samples, idx))
X = np.vstack(X)
y = np.concatenate(y)

perm = np.random.permutation(len(y))
train_size = int(0.7 * len(y))
idx_train, idx_test = perm[:train_size], perm[train_size:]
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ============================
# 2. 向量化 KNN 实现（NumPy）
# ============================
def knn_predict_vectorized(X_train, y_train, X_test, k=5, p=2, batch_size=None):
    """
    向量化计算距离，使用 argpartition 提速
    batch_size: None 表示一次性计算，否则分批
    """
    n_test, n_train = X_test.shape[0], X_train.shape[0]
    y_pred = np.empty(n_test, dtype=int)
    # 当 batch_size 未设置时，一次性计算完整距离矩阵
    if batch_size is None:
        # 计算距离矩阵: (x - y)^2 = x^2 + y^2 - 2xy
        X2 = np.sum(X_train**2, axis=1)[None, :]            # shape (1, n_train)
        T2 = np.sum(X_test**2, axis=1)[:, None]            # shape (n_test, 1)
        cross = X_test @ X_train.T                         # shape (n_test, n_train)
        dists = np.sqrt(X2 + T2 - 2 * cross)               # 欧氏距离矩阵
        # 找到 k 个最近邻的索引
        idx_k = np.argpartition(dists, k, axis=1)[:, :k]    # shape (n_test, k)
        # 投票
        for i in range(n_test):
            neighbors = y_train[idx_k[i]]
            labels, counts = np.unique(neighbors, return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
    else:
        # 分批计算以节省内存
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            Xb = X_test[start:end]
            X2 = np.sum(X_train**2, axis=1)[None, :]
            Tb2 = np.sum(Xb**2, axis=1)[:, None]
            cross = Xb @ X_train.T
            dists = np.sqrt(X2 + Tb2 - 2 * cross)
            idx_k = np.argpartition(dists, k, axis=1)[:, :k]
            for i, inds in enumerate(idx_k):
                neighbors = y_train[inds]
                labels, counts = np.unique(neighbors, return_counts=True)
                y_pred[start + i] = labels[np.argmax(counts)]
    return y_pred

# 2.1 比较向量化与循环版本速度
# 循环版本

def knn_predict_loop(X_train, y_train, X_test, k=5):
    n_test = X_test.shape[0]
    y_pred = np.empty(n_test, dtype=int)
    for i in range(n_test):
        dist = np.linalg.norm(X_train - X_test[i], axis=1)
        nn = np.argsort(dist)[:k]
        labels, counts = np.unique(y_train[nn], return_counts=True)
        y_pred[i] = labels[np.argmax(counts)]
    return y_pred

# 测速
for func in [knn_predict_loop, knn_predict_vectorized]:
    start = time.time()
    y_pred = func(X_train, y_train, X_test, k=5)
    elapsed = time.time() - start
    acc = np.mean(y_pred == y_test)
    print(f"{func.__name__}: time={elapsed:.4f}s, accuracy={acc:.3f}")

# =======================================
# 3. PCA 可视化（降到 2 维）
# =======================================
X_mean = X_train.mean(axis=0)
Xc = X_train - X_mean
cov = np.cov(Xc, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
top2 = eigvecs[:, np.argsort(eigvals)[-2:]]
X_train_pca = (X_train - X_mean) @ top2
X_test_pca = (X_test - X_mean) @ top2

plt.figure(figsize=(6,6))
colors = ['blue', 'red', 'green']
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
for cls in range(n_classes):
    plt.scatter(X_train_pca[y_train==cls,0], X_train_pca[y_train==cls,1],
                c=colors[cls], edgecolor='k', label=f'Class {cls} Train')
for cls in range(n_classes):
    plt.scatter(X_test_pca[y_test==cls,0], X_test_pca[y_test==cls,1],
                marker='x', s=80, c=colors[cls], label=f'Class {cls} Test')
plt.title('PCA Projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# =======================================
# 4. 决策边界在 PCA 空间的可视化
# =======================================
h = 0.1
x_min, x_max = X_train_pca[:,0].min() - 1, X_train_pca[:,0].max() + 1
y_min, y_max = X_train_pca[:,1].min() - 1, X_train_pca[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_orig = grid @ top2.T + X_mean
Z = knn_predict_vectorized(X_train, y_train, grid_orig, k=5)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
for cls in range(n_classes):
    plt.scatter(X_train_pca[y_train==cls,0], X_train_pca[y_train==cls,1],
                c=colors[cls], edgecolor='k', label=f'Class {cls} Train')
plt.title('Decision Boundary in PCA Space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
