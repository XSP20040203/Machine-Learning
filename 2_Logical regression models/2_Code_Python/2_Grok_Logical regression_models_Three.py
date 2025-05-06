import numpy as np
import matplotlib.pyplot as plt

# 手动生成三分类数据
def generate_data(n_samples=150, n_features=2, n_classes=3):
    #np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    # 手动创建三个类别的中心
    centers = np.array([[0, 0], [3, 3], [-3, 3]])
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        start = i * (n_samples // n_classes)
        end = (i + 1) * (n_samples // n_classes) if i < n_classes - 1 else n_samples
        y[start:end] = i
        X[start:end] += centers[i]
    return X, y

# 多分类逻辑回归（One-vs-Rest）
def multiclass_logistic_regression(X, y, n_classes, learning_rate=0.1, num_iterations=1000, threshold=0.5, plot=True):
    # Sigmoid函数
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    n_samples, n_features = X.shape
    weights = np.zeros((n_classes, n_features))  # 每类一个权重向量
    biases = np.zeros(n_classes)  # 每类一个偏置
    
    # One-vs-Rest训练
    for c in range(n_classes):
        # 将当前类标记为1，其他为0
        y_binary = (y == c).astype(int)
        
        # 训练当前类的二分类器
        for _ in range(num_iterations):
            linear_model = np.dot(X, weights[c]) + biases[c]
            y_predicted = sigmoid(linear_model)
            
            # 计算梯度并更新参数
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary))
            db = (1 / n_samples) * np.sum(y_predicted - y_binary)
            weights[c] -= learning_rate * dw
            biases[c] -= learning_rate * db
    
    # 预测
    def predict(X):
        # 计算每个类的概率
        logits = np.dot(X, weights.T) + biases
        probabilities = sigmoid(logits)
        # 返回概率最大的类别
        return np.argmax(probabilities, axis=1)
    
    # 计算准确率
    predictions = predict(X)
    accuracy = np.mean(predictions == y)
    
    # 可视化决策边界（仅适用于2D特征）
    if plot and n_features == 2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = predict(X_grid)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        for c in range(n_classes):
            plt.scatter(X[y == c][:, 0], X[y == c][:, 1], label=f'Class {c}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Multiclass Logistic Regression (One-vs-Rest)')
        plt.legend()
        plt.show()
    
    return weights, biases, accuracy

# 生成随机三分类数据
X, y = generate_data(n_samples=150, n_features=2, n_classes=3)

# 运行模型
weights, biases, accuracy = multiclass_logistic_regression(X, y, n_classes=3, learning_rate=0.1, num_iterations=1000)
print(f"Accuracy: {accuracy:.4f}")