import numpy as np
import matplotlib.pyplot as plt
class MultivariateLinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.loss_history = []  # 记录损失变化
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        self.weights = np.random.randn(n_features, n_labels) * 0.01
        self.bias = np.zeros(n_labels)
        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred - y) ** 2) / 2
            self.loss_history.append(loss)
            error = y_pred - y
            dW = np.dot(X.T, error) / n_samples
            db = np.mean(error, axis=0)
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
        return self
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def plot_loss_history(self):
        """绘制训练损失变化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.n_iter), self.loss_history, color='royalblue')
        plt.title('Training Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.grid(alpha=0.3)
        plt.show()
    def plot_predictions(self, X, y, label_index=0):
        """可视化单个标签的预测结果"""
        y_pred = self.predict(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(y[:, label_index], y_pred[:, label_index], 
                   alpha=0.6, color='crimson', edgecolors='w')
        # 绘制理想对角线
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                '--', color='gray', lw=1.5)
        plt.title(f'Label {label_index} - True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(alpha=0.3)
        plt.show()
# 示例用法
if __name__ == "__main__":
    # 生成更易可视化的带偏置数据
    np.random.seed(42)
    n_features = 3
    n_labels = 2
    # 生成具有线性可分性的数据
    X = np.random.randn(200, n_features) * 2
    print(X.shape)
    true_weights = np.array([[3.0, -1.5], [-2.0, 0.8], [1.2, -0.3]])
    true_bias = np.array([2.0, -1.0])
    # 添加带有不同量级的噪声
    y = np.dot(X, true_weights) + true_bias
    print(y.shape)
    noise = np.hstack([
        np.random.randn(200, 1)*0.5,
        np.random.randn(200, 1)*1.0
    ])
    y += noise
    # 分割训练集
    X_train, y_train = X[:180], y[:180]
    X_test, y_test = X[180:], y[180:]
    # 训练模型
    model = MultivariateLinearRegression(learning_rate=0.1, n_iter=1500)
    model.fit(X_train, y_train)
    # 可视化训练过程
    model.plot_loss_history()
    # 可视化各个标签的预测结果
    for label in range(n_labels):
        print(f"\nLabel {label} Evaluation:")
        print("True Weights:", true_weights[:, label])
        print("Learned Weights:", model.weights[:, label])
        print("True Bias:", true_bias[label])
        print("Learned Bias:", model.bias[label])
        model.plot_predictions(X_test, y_test, label_index=label)
