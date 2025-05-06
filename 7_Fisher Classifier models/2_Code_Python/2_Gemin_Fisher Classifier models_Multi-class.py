import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier # 用于在投影空间中分类并可视化决策边界
from matplotlib.colors import ListedColormap # 用于绘制决策边界颜色

# 1. 生成二维多类别数据集 (例如 3 个类别)
n_samples = 150
n_classes = 3
centers = [(-3, -3), (0, 0), (3, 3)]
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=1.5)

# 获取唯一的类别标签
unique_classes = np.unique(y)
n_features = X.shape[1] # 数据特征维度 (这里是 2D)

# 2. 实现 LDA 计算步骤

# 分离每个类别的数据
X_by_class = [X[y == c] for c in unique_classes]
N_by_class = [len(X_class) for X_class in X_by_class]

# 计算每个类别的均值向量
means = np.array([np.mean(X_class, axis=0) for X_class in X_by_class])

# 计算全局均值向量
global_mean = np.mean(X, axis=0)

# 计算类内散度矩阵 (Sw)
Sw = np.zeros((n_features, n_features))
for i, X_class in enumerate(X_by_class):
    # S_i = (X_i - m_i).T @ (X_i - m_i)
    X_minus_mi = X_class - means[i]
    Sw += X_minus_mi.T @ X_minus_mi

# 计算类间散度矩阵 (Sb)
Sb = np.zeros((n_features, n_features))
for i, m_i in enumerate(means):
    # Sb_i = N_i * (m_i - m)(m_i - m)^T
    m_i_minus_m = (m_i - global_mean).reshape(-1, 1)
    Sb += N_by_class[i] * (m_i_minus_m @ m_i_minus_m.T)

# 3. 求解广义特征值问题 S_B w = λ S_W w
# 转化为 S_W_inv @ S_B w = λ w (标准特征值问题)
try:
    Sw_inv = np.linalg.inv(Sw)
    matrix_to_solve = Sw_inv @ Sb

    # 计算特征值和特征向量
    # np.linalg.eig 返回特征值和对应的特征向量（按列存储）
    eigenvalues, eigenvectors = np.linalg.eig(matrix_to_solve)

    # 特征值和特征向量可能是复数，取实部
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 4. 选择 principal 的投影方向
    # 将特征值和特征向量配对
    eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

    # 按特征值大小降序排序
    eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)

    # LDA 最多找到 min(特征维度, 类别数 - 1) 个有用的投影方向
    # 这里是 min(2, 3 - 1) = 2
    n_components = min(n_features, n_classes - 1)

    # 选择前 n_components 个特征向量作为投影矩阵 W
    W = np.array([pair[1] for pair in eigenpairs[:n_components]]).T # .T 确保特征向量是列向量

    print(f"原始特征维度: {n_features}")
    print(f"类别数量: {n_classes}")
    print(f"LDA 投影到维度: {n_components}")
    print(f"投影矩阵 W 的形状: {W.shape}")

    # 5. 将数据投影到 W 决定的子空间上
    # Y = X @ W
    Y = X @ W

    # 7. 在投影后的低维空间中可视化数据
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, unique_classes, unique_classes):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=f'Class {target_name}')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.axis('equal')
    plt.grid(True)


    plt.subplot(1, 2, 2)
    # 投影到 2D (因为原始数据是 2D, n_components=2)
    if n_components == 2:
        for color, i, target_name in zip(colors, unique_classes, unique_classes):
            plt.scatter(Y[y == i, 0], Y[y == i, 1], alpha=.8, color=color,
                        label=f'Class {target_name}')
        plt.title('Data Projected onto LDA Subspace (2D)')
        plt.xlabel('LDA Component 1')
        plt.ylabel('LDA Component 2')
    # 如果投影到 1D (例如原始数据是 3D, k=3, 投影到 2D, 如果只想看第一主成分)
    elif n_components == 1:
         for color, i, target_name in zip(colors, unique_classes, unique_classes):
            plt.scatter(Y[y == i], np.zeros_like(Y[y == i]), alpha=.8, color=color,
                        label=f'Class {target_name}')
         plt.title('Data Projected onto LDA Component 1 (1D)')
         plt.xlabel('LDA Component 1')
         plt.ylabel('') # No ylabel for 1D projection plot
         plt.yticks([]) # Hide y-axis ticks for 1D plot
    else:
         print(f"投影维度为 {n_components}，此代码仅可视化 1D 或 2D 投影。")

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.grid(True)


    # 8. 在原始空间中绘制基于投影后数据的分类决策边界
    # 使用一个简单的分类器（如 KNN）在投影空间中训练
    # 然后在原始空间的网格上预测，再将网格点投影，用训练好的分类器预测投影点
    # 最后根据预测结果绘制决策边界

    if n_components >= 1: # Only sensible to draw boundary if projected to at least 1D
        # 训练一个分类器（例如 KNN）在投影后的数据 Y 上
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(Y, y)

        # 创建原始空间的网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # 将网格点投影到 LDA 子空间
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        projected_grid_points = grid_points @ W

        # 在投影空间中对网格点进行分类
        Z = classifier.predict(projected_grid_points)
        Z = Z.reshape(xx.shape)

        # 绘制决策边界的颜色区域
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) # 浅色
        # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # 深色 - 用于点

        plt.subplot(1, 2, 1) # 在原始数据图上绘制边界
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3) # 绘制决策区域背景色

        # 重新绘制原始数据点以显示在背景色之上
        for color, i, target_name in zip(colors, unique_classes, unique_classes):
            plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                        label=f'Class {target_name}')

        plt.title('Original Data with LDA-based Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.axis('equal') # 保持x和y轴比例一致
        plt.grid(True)


    plt.tight_layout()
    plt.show()

except np.linalg.LinAlgError:
    print("警告：类内散度矩阵不可逆。可能因为数据维度过高、样本过少或某些类别样本点完全重合。无法计算 LDA。")