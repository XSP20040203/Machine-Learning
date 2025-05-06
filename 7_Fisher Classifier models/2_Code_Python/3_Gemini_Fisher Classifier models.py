import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # 用于分割训练集和测试集
from sklearn.metrics import accuracy_score # 用于评估分类准确率
from matplotlib.colors import ListedColormap

# 1. 生成多维多类别数据集 (例如 4 个特征, 3 个类别)
n_samples = 200 # 增加样本数
n_features = 4 # 特征维度
n_classes = 3
# 随机生成类别中心，以创建一些分离度
centers = np.random.rand(n_classes, n_features) * 10 - 5 # 范围在 -5 到 5
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=2.0, n_features=n_features)

print(f"原始数据形状: {X.shape}")
print(f"类别标签形状: {y.shape}")
print(f"类别数量: {np.unique(y)}")

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # stratify 确保训练集和测试集类别比例相似

# 获取训练集中的唯一类别标签
unique_classes = np.unique(y_train)
n_train_samples = X_train.shape[0]

# 2. 实现 LDA 计算步骤 (在训练集上计算参数)

# 分离训练集数据 by class
X_train_by_class = [X_train[y_train == c] for c in unique_classes]
N_train_by_class = [len(X_class) for X_class in X_train_by_class]

# 计算每个类别的均值向量
means_train = np.array([np.mean(X_class, axis=0) for X_class in X_train_by_class])

# 计算全局均值向量
global_mean_train = np.mean(X_train, axis=0)

# 计算类内散度矩阵 (Sw)
Sw = np.zeros((n_features, n_features))
for i, X_class in enumerate(X_train_by_class):
    X_minus_mi = X_class - means_train[i]
    Sw += X_minus_mi.T @ X_minus_mi # (N_i, d).T @ (N_i, d) = (d, N_i) @ (N_i, d) = (d, d)

# 计算类间散度矩阵 (Sb)
Sb = np.zeros((n_features, n_features))
for i, m_i in enumerate(means_train):
    m_i_minus_m = (m_i - global_mean_train).reshape(-1, 1)
    Sb += N_train_by_class[i] * (m_i_minus_m @ m_i_minus_m.T) # (d, 1) @ (1, d) = (d, d)

# 3. 求解广义特征值问题 S_B w = λ S_W w
try:
    # 检查 Sw 是否可逆，如果不可逆，可能需要进行正则化或使用伪逆
    # 简单起见，这里直接尝试求逆
    Sw_inv = np.linalg.inv(Sw)
    matrix_to_solve = Sw_inv @ Sb

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix_to_solve)

    # 取实部
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 4. 选择 principal 的投影方向
    eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)

    # LDA 最多找到 min(特征维度, 类别数 - 1) 个有用的投影方向
    n_components = min(n_features, n_classes - 1)
    print(f"LDA 投影到维度: {n_components}")

    # 选择前 n_components 个特征向量作为投影矩阵 W
    W = np.array([pair[1] for pair in eigenpairs[:n_components]]).T # (d, p) 形状

    # 5. 将数据投影到 W 决定的子空间上
    # 投影训练集和测试集
    Y_train = X_train @ W # (N_train, d) @ (d, p) = (N_train, p)
    Y_test = X_test @ W   # (N_test, d) @ (d, p) = (N_test, p)

    print(f"训练集投影后形状: {Y_train.shape}")
    print(f"测试集投影后形状: {Y_test.shape}")


    # 6. 在投影后的空间中进行分类

    if n_components >= 1:
        # 使用 KNN 分类器在投影后的训练数据上训练
        # 我们可以通过调整 n_neighbors 来看效果
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(Y_train, y_train)

        # 在投影后的测试数据上进行预测
        y_pred_test = classifier.predict(Y_test)

        # 评估分类准确率
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"在投影后的 {n_components} 维空间上使用 KNN 的测试集准确率: {accuracy:.4f}")

        # 7. 可视化投影后的数据 (如果投影到 1D, 2D 或 3D)

        plt.figure(figsize=(10, 8))
        colors = ['navy', 'turquoise', 'darkorange', 'green', 'red'] # 增加一些颜色
        lw = 2

        if n_components == 1:
            # 投影到 1D
            for color, i, target_name in zip(colors, unique_classes, unique_classes):
                plt.scatter(Y_test[y_test == i], np.zeros_like(Y_test[y_test == i]), alpha=.8, color=color,
                            label=f'Class {target_name} (Test)')
            plt.title('Test Data Projected onto LDA Component 1 (1D)')
            plt.xlabel('LDA Component 1')
            plt.ylabel('')
            plt.yticks([]) # Hide y-axis ticks for 1D plot
            plt.grid(True, axis='x')

        elif n_components == 2:
            # 投影到 2D
            for color, i, target_name in zip(colors, unique_classes, unique_classes):
                plt.scatter(Y_test[y_test == i, 0], Y_test[y_test == i, 1], alpha=.8, color=color,
                            label=f'Class {target_name} (Test)')

            # 在投影空间中绘制 KNN 的决策边界（仅在 2D 投影时方便）
            y_min, y_max = Y_test[:, 1].min() - 1, Y_test[:, 1].max() + 1
            x_min, x_max = Y_test[:, 0].min() - 1, Y_test[:, 0].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))

            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF', '#FFDAB9']) # 对应类别颜色
            plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)


            plt.title('Test Data Projected onto LDA Subspace (2D)')
            plt.xlabel('LDA Component 1')
            plt.ylabel('LDA Component 2')
            plt.grid(True)


        elif n_components == 3:
             # 投影到 3D
             fig = plt.figure(figsize=(10, 8))
             ax = fig.add_subplot(111, projection='3d')
             for color, i, target_name in zip(colors, unique_classes, unique_classes):
                 ax.scatter(Y_test[y_test == i, 0], Y_test[y_test == i, 1], Y_test[y_test == i, 2], alpha=.8, color=color,
                            label=f'Class {target_name} (Test)')
             ax.set_title('Test Data Projected onto LDA Subspace (3D)')
             ax.set_xlabel('LDA Component 1')
             ax.set_ylabel('LDA Component 2')
             ax.set_zlabel('LDA Component 3')
             ax.legend(loc='best', shadow=False, scatterpoints=1)
             plt.grid(True)

        else:
            print(f"投影维度为 {n_components}，此代码仅可视化 1D, 2D 或 3D 投影后的数据。")


        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.suptitle(f'LDA Projection Results (Original Features: {n_features}D, Classes: {n_classes})', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    else:
         print("投影维度为 0，无法进行分类和可视化。")


except np.linalg.LinAlgError:
    print("警告：类内散度矩阵不可逆。可能因为数据维度过高、样本过少或某些类别样本点在某些维度上完全重合。无法计算 LDA。")
    print("尝试增加样本数量，或检查数据是否存在完全线性相关的特征。")

# 可选：对比一下在原始高维空间直接用 KNN 的准确率 (没有可视化原始空间的决策边界，因为它很难)
# print("\n--- 对比 ---")
# classifier_original = KNeighborsClassifier(n_neighbors=5)
# classifier_original.fit(X_train, y_train)
# y_pred_original_test = classifier_original.predict(X_test)
# accuracy_original = accuracy_score(y_test, y_pred_original_test)
# print(f"在原始 {n_features} 维空间上直接使用 KNN 的测试集准确率: {accuracy_original:.4f}")