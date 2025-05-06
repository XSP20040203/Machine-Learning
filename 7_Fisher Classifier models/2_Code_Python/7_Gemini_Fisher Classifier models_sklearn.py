import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting if n_components is 3
from matplotlib.colors import ListedColormap # For plotting decision boundaries

# --- 1. 数据生成 ---
n_samples = 300
n_features = 5    # 原始特征数量 (多于2个)
n_informative = 3 # 其中有用的特征数量
n_classes = 4     # 类别数量 (多于2个)
# make_classification 创建的数据通常比 make_blobs 更复杂，更适合测试分类器
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           n_redundant=1, n_repeated=1, n_classes=n_classes, n_clusters_per_class=1,
                           weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, random_state=42)

print(f"原始数据形状: {X.shape}")
print(f"类别标签形状: {y.shape}")
print(f"类别数量: {np.unique(y)}")

# --- 2. 数据分割 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# --- 3. 使用 scikit-learn 的 LinearDiscriminantAnalysis 直接作为分类器 ---
print("\n--- LDA 直接作为分类器 ---")
# n_components=None 或不设置时，LDA 作为分类器会使用所有判别方向（最多 min(d, k-1)）
lda_classifier_direct = LinearDiscriminantAnalysis()
lda_classifier_direct.fit(X_train, y_train)
y_pred_lda_direct = lda_classifier_direct.predict(X_test)
accuracy_lda_direct = accuracy_score(y_test, y_pred_lda_direct)
print(f"LDA (直接分类器) 测试集准确率: {accuracy_lda_direct:.4f}")

# --- 4. 使用 LDA 进行降维 (特征提取) 并结合其他分类器 ---
# LDA 投影到的维度最多是 min(原始特征维度, 类别数 - 1)
n_components_lda = min(X_train.shape[1], len(np.unique(y_train)) - 1)
print(f"\nLDA 投影维度 (理论最大): {n_components_lda}")

# 为了可视化方便，如果理论最大维度大于 3，我们可以强制设置为 2 或 3 进行可视化
# 但请注意，这仅用于可视化，性能评估仍应使用 n_components_lda
n_components_viz = n_components_lda
if n_components_viz > 3:
    print(f"警告: 理论投影维度 {n_components_lda}D 大于 3D，将仅用于性能评估，跳过投影数据可视化。")
    n_components_viz = 0 # 标记不进行可视化

# 创建 LDA Transformer (设置 n_components 用于降维)
# 我们使用理论最大维度进行实际的性能评估
lda_transformer = LinearDiscriminantAnalysis(n_components=n_components_lda)

# 定义不同的分类器
classifiers = {
    "KNN (n_neighbors=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000), # 增加迭代次数防止收敛警告
    "SVC (Linear Kernel)": SVC(kernel='linear', probability=True), # probability=True 可以帮助绘制决策边界
    # "SVC (RBF Kernel)": SVC(kernel='rbf') # RBF 核 SVM 通常不需要先进行线性降维
}

print("\n--- LDA 降维 + 其他分类器 (使用 Pipeline) ---")

results = {}
# 存储用于可视化（如果维度合适）的 Pipeline 的投影测试数据和对应的分类器
projected_test_data_for_viz = None
classifier_for_viz = None
viz_classifier_name = None
lda_transformer_for_viz = None # 用于可视化的 LDA transformer (如果 n_components_lda > 3 但 n_components_viz = 2 or 3)


# 如果实际用于可视化的维度小于理论最大维度，需要创建一个单独的 LDA Transformer 用于可视化
if n_components_viz > 0 and n_components_viz < n_components_lda:
     lda_transformer_for_viz = LinearDiscriminantAnalysis(n_components=n_components_viz).fit(X_train, y_train)


for name, classifier in classifiers.items():
    # 创建 Pipeline: 先 LDA 降维，后接分类器 (使用理论最大维度进行性能评估)
    pipeline = Pipeline([
        ('lda', lda_transformer),
        ('classifier', classifier)
    ])

    # 在训练数据上训练 Pipeline
    pipeline.fit(X_train, y_train)

    # 在测试数据上进行预测和评估
    y_pred_pipeline = pipeline.predict(X_test)
    accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
    results[name] = accuracy_pipeline

    print(f"Pipeline: LDA ({n_components_lda}D) + {name} 测试集准确率: {accuracy_pipeline:.4f}")

    # 如果需要可视化 (即 n_components_viz > 0)，并且还没有存储可视化数据，则保留当前 Pipeline 的测试集投影结果和分类器
    if n_components_viz > 0 and projected_test_data_for_viz is None:
         if lda_transformer_for_viz is not None:
              # 如果使用单独的 LDA Transformer for viz
              projected_test_data_for_viz = lda_transformer_for_viz.transform(X_test)
              # 在用于可视化的投影数据上重新训练分类器
              classifier_for_viz = classifier.fit(lda_transformer_for_viz.transform(X_train), y_train)
         else:
             # 如果理论最大维度就适合可视化 (n_components_lda == n_components_viz)
             projected_test_data_for_viz = pipeline.named_steps['lda'].transform(X_test)
             # 获取分类器 step (它已经在 n_components_lda 维空间训练好了)
             classifier_for_viz = pipeline.named_steps['classifier']

         viz_classifier_name = name


# --- 5. 模型可视化: 绘制投影后的测试数据和决策边界 ---
# 注意：这里可视化的是数据投影到 LDA 低维空间后的分布，以及训练在该低维空间上的分类器的决策边界。
# 它不直接表示原始高维空间中的决策边界。
if n_components_viz > 0 and projected_test_data_for_viz is not None and classifier_for_viz is not None:
    print(f"\n--- 可视化投影后的测试数据 ({n_components_viz}D) 和决策边界 ---")

    fig = plt.figure(figsize=(9, 7))
    colors = ['navy', 'turquoise', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    unique_plot_classes = np.unique(y_test) # 使用测试集的类别
    # 为每个类别映射一个颜色
    class_color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_plot_classes)}


    if n_components_viz == 2:
        ax = fig.add_subplot(111)
        # 绘制投影后的测试数据点
        for i in unique_plot_classes:
            ax.scatter(projected_test_data_for_viz[y_test == i, 0], projected_test_data_for_viz[y_test == i, 1],
                       alpha=.8, color=class_color_map[i], label=f'Class {i} (Test)')

        # 在 2D 投影空间绘制分类器的决策边界
        # 创建投影空间的网格
        x_min, x_max = projected_test_data_for_viz[:, 0].min() - 1, projected_test_data_for_viz[:, 0].max() + 1
        y_min, y_max = projected_test_data_for_viz[:, 1].min() - 1, projected_test_data_for_viz[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # 预测网格点类别
        # 需要确保分类器有 predict 方法
        if hasattr(classifier_for_viz, 'predict'):
            # 注意：有些分类器（如 SVC）的 predict 方法在概率模式下可能需要形状特殊的输入
            # 但对于 KNN, Logistic Regression 等，np.c_ 就足够了
            try:
                 Z = classifier_for_viz.predict(np.c_[xx.ravel(), yy.ravel()])
                 Z = Z.reshape(xx.shape)

                 # 创建对应类别的颜色映射，确保 ListedColormap 的颜色顺序与 Z 中的类别标签顺序一致
                 plot_classes_in_Z = np.unique(Z) # Z 中实际出现的类别
                 cmap_light_colors = [class_color_map[cls] for cls in plot_classes_in_Z]
                 cmap_light = ListedColormap([c + '33' for c in cmap_light_colors]) # 更浅的颜色

                 ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
            except Exception as e:
                 print(f"警告: 绘制 {viz_classifier_name} 决策边界时出错: {e}")


        ax.set_title(f'Test Data Projected onto LDA Subspace ({n_components_viz}D) \n and {viz_classifier_name} Decision Boundary')
        ax.set_xlabel(f'LDA Component 1 (Eigenvalue: {lda_transformer.explained_variance_ratio_[0]:.2f})') # 可以显示特征值解释的方差比例
        ax.set_ylabel(f'LDA Component 2 (Eigenvalue: {lda_transformer.explained_variance_ratio_[1]:.2f})')


    elif n_components_viz == 3:
         ax = fig.add_subplot(111, projection='3d')
         for i in unique_plot_classes:
             ax.scatter(projected_test_data_for_viz[y_test == i, 0], projected_test_data_for_viz[y_test == i, 1], projected_test_data_for_viz[y_test == i, 2],
                        alpha=.8, color=class_color_map[i], label=f'Class {i} (Test)')
         ax.set_title('Test Data Projected onto LDA Subspace (3D)')
         ax.set_xlabel(f'LDA Component 1 (Eigenvalue: {lda_transformer.explained_variance_ratio_[0]:.2f})')
         ax.set_ylabel(f'LDA Component 2 (Eigenvalue: {lda_transformer.explained_variance_ratio_[1]:.2f})')
         ax.set_zlabel(f'LDA Component 3 (Eigenvalue: {lda_transformer.explained_variance_ratio_[2]:.2f})')

    if fig is not None:
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.grid(True)
        plt.show()


# --- 6. 直接在原始高维空间使用分类器进行对比 ---
print("\n--- 直接在原始高维空间使用分类器 ---")
classifiers_original = {
    "KNN (n_neighbors=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC (Linear Kernel)": SVC(kernel='linear'),
    # "SVC (RBF Kernel)": SVC(kernel='rbf') # RBF 核 SVM 通常可以直接在高维空间表现良好
}

results_original = {}
for name, classifier in classifiers_original.items():
    classifier.fit(X_train, y_train)
    y_pred_original = classifier.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    results_original[name] = accuracy_original
    print(f"{name} (原始 {n_features}D) 测试集准确率: {accuracy_original:.4f}")

# --- 7. 总结准确率 ---
print("\n--- 准确率总结 ---")
print(f"LDA (直接分类器): {accuracy_lda_direct:.4f}")
print(f"LDA 降维 ({n_components_lda}D) 后 + 其他分类器:")
for name, acc in results.items():
    print(f"  {name}: {acc:.4f}")
print(f"原始高维空间 ({n_features}D) 直接使用分类器:")
for name, acc in results_original.items():
     print(f"  {name}: {acc:.4f}")