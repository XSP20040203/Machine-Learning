import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 1. 生成模拟数据
np.random.seed(42)
n1, n2 = 50, 50
# 两类的均值和协方差
mean1 = np.array([2, 2])
mean2 = np.array([5, 5])
cov = np.array([[1.0, 0.5], [0.5, 1.0]])

X1 = np.random.multivariate_normal(mean1, cov, n1)
X2 = np.random.multivariate_normal(mean2, cov, n2)
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(n1), np.ones(n2)))  # 0 表示类1, 1 表示类2

# 2. 计算类内散度矩阵 S_w
m1 = X1.mean(axis=0)
m2 = X2.mean(axis=0)

S1 = np.dot((X1 - m1).T, (X1 - m1))
S2 = np.dot((X2 - m2).T, (X2 - m2))
Sw = S1 + S2

# 3. 计算最优投影方向 w
w = np.linalg.inv(Sw).dot(m2 - m1)
w = w / np.linalg.norm(w)  # 单位化方向

# 4. 投影数据到一维
proj = X.dot(w)
proj1 = proj[y == 0]
proj2 = proj[y == 1]

# 5. 决策阈值
threshold = 0.5 * (proj1.mean() + proj2.mean())

# 6. 分类预测
y_pred = (proj >= threshold).astype(int)

# 7. 可视化原始空间和投影结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 7.1 原始空间散点图与投影方向线
axes[0].scatter(X1[:, 0], X1[:, 1], label='Class 1')
axes[0].scatter(X2[:, 0], X2[:, 1], label='Class 2')
# 绘制投影方向
origin = np.mean(X, axis=0)
line = np.vstack((origin - 5*w, origin + 5*w))
axes[0].plot(line[:, 0], line[:, 1], 'k--', label='LDA direction')
axes[0].set_title('原始空间与投影方向')
axes[0].legend()

# 7.2 投影后直方图与阈值
axes[1].hist(proj1, bins=15, alpha=0.7, label='Class 1')
axes[1].hist(proj2, bins=15, alpha=0.7, label='Class 2')
axes[1].axvline(threshold, color='k', linestyle='--', label='Decision threshold')
axes[1].set_title('投影后的一维分布')
axes[1].legend()

plt.tight_layout()
plt.show()
