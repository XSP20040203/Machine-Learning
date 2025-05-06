import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据集
np.random.seed(42)
num_samples = 100

# 类别0的数据 (中心在[1,1])
X0 = np.random.randn(num_samples, 2) * 0.5 + [1, 1]
y0 = np.zeros(num_samples)

# 类别1的数据 (中心在[3,3])
X1 = np.random.randn(num_samples, 2) * 0.5 + [3, 3]
y1 = np.ones(num_samples)

# 合并数据集
X = np.vstack((X0, X1))
y = np.hstack((y0, y1)).reshape(-1, 1)

# 打乱数据集
shuffle_idx = np.random.permutation(2 * num_samples)
X = X[shuffle_idx]
y = y[shuffle_idx]

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 初始化参数
w = np.zeros((2, 1))
b = 0.0

# 超参数设置
learning_rate = 0.01
num_epochs = 100000

# 训练过程记录
losses = []

# 训练逻辑回归模型
for epoch in range(num_epochs):
    # 前向传播
    z = X.dot(w) + b
    a = sigmoid(z)
    
    # 添加极小值避免log(0)
    a = np.clip(a, 1e-15, 1 - 1e-15)
    
    # 计算交叉熵损失
    loss = -(y * np.log(a) + (1 - y) * np.log(1 - a)).mean()
    losses.append(loss)
    
    # 反向传播计算梯度
    dw = X.T.dot(a - y) / X.shape[0]
    db = (a - y).mean()
    
    # 参数更新
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # 每100次迭代打印损失
    if epoch % 1000 == 0:
        print(f'Epoch {epoch:4d}, Loss: {loss:.4f}')

# 绘制训练损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.grid(True)
plt.legend(['Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')

# 绘制决策边界
plt.subplot(1, 2, 2)
# 绘制原始数据
plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 0')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1')

# 生成决策边界数据点
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x_values = np.array([x1_min, x1_max])
y_values = (- (w[0] * x_values + b )) / w[1]

plt.plot(x_values, y_values, color='green', lw=2, 
         label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.legend()
plt.show()
