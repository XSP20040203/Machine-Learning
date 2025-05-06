import numpy as np # 逻辑回归模型示例
import matplotlib.pyplot as plt # 导入必要的库

# Sigmoid函数
def sigmoid(z):
    z = np.clip(z, -500, 500) # 避免溢出
    return 1 / (1 + np.exp(-z)) # Sigmoid函数的导数

# 训练逻辑回归模型
def train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000):
    n_samples, n_features = X.shape # 获取样本数和特征数
    
    # 初始化参数
    weights = np.zeros(n_features) # 权重初始化为0
    bias = 0 # 偏置初始化为0
    
    # 梯度下降
    for _ in range(num_iterations):
        linear_model = np.dot(X, weights) + bias # 线性模型
        y_predicted = sigmoid(linear_model) # Sigmoid函数输出
        
        # 计算梯度
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # 权重的梯度
        db = (1 / n_samples) * np.sum(y_predicted - y) # 偏置的梯度
        
        # 更新参数
        weights = weights - learning_rate * dw # 更新权重
        bias = bias - learning_rate * db # 更新偏置
    
    return weights, bias

# 预测概率
def predict_proba(X, weights, bias):
    linear_model = np.dot(X, weights) + bias # 线性模型
    return sigmoid(linear_model)

# 预测类别
def predict(X, weights, bias, threshold=0.5):
    probabilities = predict_proba(X, weights, bias) # 计算概率
    return (probabilities >= threshold).astype(int) 

# 可视化决策边界
def plot_decision_boundary(X, y, weights, bias):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # 获取特征1的范围
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # 获取特征2的范围
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1)) # 创建网格
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(X_grid, weights, bias)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.show()

# 生成随机数据
np.random.seed(0) # 设置随机种子
X = np.random.randn(100, 2) # 生成100个样本，2个特征
y = (X[:, 0] + X[:, 1] > 0).astype(int) # 生成标签

# 训练模型
weights, bias = train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000)

# 预测
predictions = predict(X, weights, bias) # 预测类别
accuracy = np.mean(predictions == y) # 计算准确率
print(f"Accuracy: {accuracy:.4f}")

# 可视化
plot_decision_boundary(X, y, weights, bias)