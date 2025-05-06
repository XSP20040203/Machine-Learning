import numpy as np
import matplotlib.pyplot as plt

# 1. 生成合成数据集
np.random.seed(0)
# 两类分布：圆心为(-2, 0)与(2, 0)
n_samples = 100
X1 = np.random.randn(n_samples, 2) + np.array([-2, 0])
X2 = np.random.randn(n_samples, 2) + np.array([2, 0])
X = np.vstack((X1, X2))
y = np.array([0]*n_samples + [1]*n_samples)

# 数据集可视化
plt.figure(figsize=(6,6))
plt.scatter(X1[:,0], X1[:,1], c='blue', label='Class 0')
plt.scatter(X2[:,0], X2[:,1], c='red', label='Class 1')
plt.title('Dataset Visualization')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 2. 实现 KNN 算法（numpy 版）
def knn_predict(X_train, y_train, X_test, k=5, p=2):
    """
    使用 Minkowski 距离（p范数）实现 KNN 分类
    X_train: 训练样本，shape=(n_train, d)
    y_train: 训练标签，shape=(n_train,)
    X_test: 测试样本，shape=(n_test, d)
    k: 最近邻个数
    p: Minkowski 距离范数
    返回: 预测标签，shape=(n_test,)
    """
    n_test = X_test.shape[0]
    y_pred = np.empty(n_test, dtype=int)

    # 对每个测试点，计算距离并投票
    for i in range(n_test):
        # 计算与所有训练点的距离
        dist = np.linalg.norm(X_train - X_test[i], ord=p, axis=1)
        # 找到 k 个最小距离的索引
        nn_idx = np.argsort(dist)[:k]
        # 多数投票
        votes = y_train[nn_idx]
        # 统计票数最多的类别
        labels, counts = np.unique(votes, return_counts=True)
        y_pred[i] = labels[np.argmax(counts)]
    return y_pred

# 3. 可视化模型决策边界
# 构建网格
h = 0.1  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# 网格预测
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn_predict(X, y, grid_points, k=5)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X1[:,0], X1[:,1], c='blue', edgecolor='k', label='Class 0')
plt.scatter(X2[:,0], X2[:,1], c='red', edgecolor='k', label='Class 1')
plt.title('KNN Decision Boundary (k=5)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 4. 在新测试点上做分类预测并可视化
# 随机生成一些测试点并预测
X_test = np.random.randn(30, 2) * 1.5
y_test_pred = knn_predict(X, y, X_test, k=5)

plt.figure(figsize=(6,6))
# 绘制训练数据
plt.scatter(X1[:,0], X1[:,1], c='blue', label='Class 0 Train')
plt.scatter(X2[:,0], X2[:,1], c='red', label='Class 1 Train')
# 绘制测试点，按预测类别上色
plt.scatter(X_test[y_test_pred==0,0], X_test[y_test_pred==0,1],
            c='cyan', marker='x', s=100, label='Predicted 0')
plt.scatter(X_test[y_test_pred==1,0], X_test[y_test_pred==1,1],
            c='magenta', marker='x', s=100, label='Predicted 1')
plt.title('Test Points Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
