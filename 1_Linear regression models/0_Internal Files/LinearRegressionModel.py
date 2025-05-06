import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 解决 Matplotlib 中文显示问题（如果需要）
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class LinearRegressionModel:
    """
    线性回归模型封装类 (纯NumPy实现)
    支持二维和高维特征训练、评估和可视化
    """
    
    def __init__(self, classification_threshold=0.5, learning_rate=0.01, max_iterations=10000, tol=1e-6):
        """
        初始化线性回归模型
        
        参数:
        classification_threshold - 分类任务的阈值，默认为0.5
        learning_rate - 学习率，默认0.01
        max_iterations - 最大迭代次数，默认1000
        tol - 停止条件的容差值，默认1e-6
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.threshold = classification_threshold
        self.weights = None
        self.bias = None
        self.is_fitted = False
        self.history = {'loss': []}
        self.feature_dim = None
        
    def _standardize(self, X):
        """
        标准化特征
        
        参数:
        X - 输入特征数组
        
        返回:
        标准化后的特征数组
        """
        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # 防止除以零
            self.std = np.where(self.std == 0, 1, self.std)
            
        return (X - self.mean) / self.std
    
    def _add_bias(self, X):
        """
        为特征矩阵添加偏置项
        
        参数:
        X - 输入特征数组
        
        返回:
        添加了偏置项的特征数组
        """
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def fit(self, X, y, test_size=0.2, random_state=42):
        """
        训练模型
        
        参数:
        X - 输入特征，可以是二维或高维
        y - 目标变量
        test_size - 测试集比例，默认0.2
        random_state - 随机种子，默认42
        
        返回:
        self - 模型实例
        """
        # 确保X是二维数组
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.feature_dim = X.shape[1]
        
        # 设置随机种子
        np.random.seed(random_state)
        
        # 划分训练测试集
        indices = np.random.permutation(X.shape[0])
        test_size_int = int(X.shape[0] * test_size)
        test_indices = indices[:test_size_int]
        train_indices = indices[test_size_int:]
        
        self.X_train, self.X_test = X[train_indices], X[test_indices]
        self.y_train, self.y_test = y[train_indices], y[test_indices]
        
        # 标准化特征
        self.X_train_scaled = self._standardize(self.X_train)
        self.X_test_scaled = self._standardize(self.X_test)
        
        # 初始化权重和偏置
        self.weights = np.zeros(self.feature_dim)
        self.bias = 0
        
        # 梯度下降训练
        m = self.X_train_scaled.shape[0]  # 样本数量
        
        for iteration in range(self.max_iterations):
            # 计算预测值
            y_pred = np.dot(self.X_train_scaled, self.weights) + self.bias
            
            # 计算梯度
            dw = (1/m) * np.dot(self.X_train_scaled.T, (y_pred - self.y_train))
            db = (1/m) * np.sum(y_pred - self.y_train)
            
            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 计算损失
            loss = self._compute_mse(self.y_train, y_pred)
            self.history['loss'].append(loss)
            
            # 收敛检查
            if iteration > 0 and abs(self.history['loss'][iteration-1] - loss) < self.tol:
                break
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        预测函数
        
        参数:
        X - 输入特征
        
        返回:
        预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        X_scaled = self._standardize(X)
        return np.dot(X_scaled, self.weights) + self.bias
    
    def predict_class(self, X):
        """
        分类预测函数 (将回归结果二值化)
        
        参数:
        X - 输入特征
        
        返回:
        二值化的分类结果
        """
        y_pred = self.predict(X)
        return (y_pred >= self.threshold).astype(int)
    
    def _compute_mse(self, y_true, y_pred):
        """
        计算均方误差
        
        参数:
        y_true - 真实值
        y_pred - 预测值
        
        返回:
        均方误差
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_r2(self, y_true, y_pred):
        """
        计算R²分数
        
        参数:
        y_true - 真实值
        y_pred - 预测值
        
        返回:
        R²分数
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def _compute_confusion_matrix(self, y_true, y_pred):
        """
        计算混淆矩阵
        
        参数:
        y_true - 真实值
        y_pred - 预测值
        
        返回:
        2x2混淆矩阵 (TN, FP, FN, TP)
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        return np.array([[TN, FP], [FN, TP]])
    
    def _compute_accuracy(self, y_true, y_pred):
        """
        计算准确率
        
        参数:
        y_true - 真实值
        y_pred - 预测值
        
        返回:
        准确率
        """
        return np.mean(y_true == y_pred)
    
    def _compute_precision(self, y_true, y_pred):
        """
        计算精确率
        
        参数:
        y_true - 真实值
        y_pred - 预测值
        
        返回:
        精确率
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        
        if TP + FP == 0:
            return 0
        return TP / (TP + FP)
    
    def _compute_recall(self, y_true, y_pred):
        """
        计算召回率
        
        参数:
        y_true - 真实值
        y_pred - 预测值
        
        返回:
        召回率
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        if TP + FN == 0:
            return 0
        return TP / (TP + FN)
    
    def _compute_f1(self, precision, recall):
        """
        计算F1分数
        
        参数:
        precision - 精确率
        recall - 召回率
        
        返回:
        F1分数
        """
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def _compute_roc(self, y_true, y_scores):
        """
        计算ROC曲线数据
        
        参数:
        y_true - 真实二元标签
        y_scores - 预测分数
        
        返回:
        fpr - 假正例率
        tpr - 真正例率
        thresholds - 阈值
        """
        # 对预测分数进行排序并找出唯一值作为阈值
        thresholds = np.unique(y_scores)
        thresholds = np.insert(thresholds, 0, thresholds[0] - 1)
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        
        m = len(thresholds)
        tpr = np.zeros(m)
        fpr = np.zeros(m)
        
        for i, threshold in enumerate(thresholds):
            y_pred = (y_scores >= threshold).astype(int)
            
            # 真正例率和假正例率
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            TN = np.sum((y_true == 0) & (y_pred == 0))
            FN = np.sum((y_true == 1) & (y_pred == 0))
            
            tpr[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
            fpr[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        # 确保单调递增
        indices = np.argsort(fpr)
        fpr = fpr[indices]
        tpr = tpr[indices]
        thresholds = thresholds[indices]
        
        return fpr, tpr, thresholds
    
    def _compute_auc(self, fpr, tpr):
        """
        计算AUC值
        
        参数:
        fpr - 假正例率
        tpr - 真正例率
        
        返回:
        auc - 曲线下面积
        """
        # 使用梯形法则计算AUC
        return np.trapz(tpr, fpr)
    
    def _compute_pr_curve(self, y_true, y_scores):
        """
        计算精确率-召回率曲线
        
        参数:
        y_true - 真实二元标签
        y_scores - 预测分数
        
        返回:
        precision - 精确率
        recall - 召回率
        thresholds - 阈值
        """
        # 对预测分数进行排序并找出唯一值作为阈值
        thresholds = np.unique(y_scores)
        thresholds = np.insert(thresholds, 0, thresholds[0] - 1)
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        
        m = len(thresholds)
        precision = np.zeros(m)
        recall = np.zeros(m)
        
        for i, threshold in enumerate(thresholds):
            y_pred = (y_scores >= threshold).astype(int)
            
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))
            
            precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 1  # 当没有正例预测时，精确率定义为1
            recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # 确保单调递减
        indices = np.argsort(recall)
        recall = recall[indices]
        precision = precision[indices]
        thresholds = thresholds[indices]
        
        return precision, recall, thresholds
        
    def evaluate(self, plot=True):
        """
        评估模型并可视化结果
        
        参数:
        plot - 是否绘制可视化图表，默认True
        
        返回:
        包含评估指标的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 回归预测
        y_pred = self.predict(self.X_test)
        
        # 回归评估指标
        mse = self._compute_mse(self.y_test, y_pred)
        r2 = self._compute_r2(self.y_test, y_pred)
        
        # 对于二分类任务
        y_pred_binary = (y_pred >= self.threshold).astype(int)
        y_test_binary = (self.y_test >= self.threshold).astype(int)
        
        # 计算混淆矩阵
        cm = self._compute_confusion_matrix(y_test_binary, y_pred_binary)
        
        # 分类评估指标
        accuracy = self._compute_accuracy(y_test_binary, y_pred_binary)
        precision = self._compute_precision(y_test_binary, y_pred_binary)
        recall = self._compute_recall(y_test_binary, y_pred_binary)
        f1 = self._compute_f1(precision, recall)
            
        # 计算ROC曲线数据
        fpr, tpr, _ = self._compute_roc(y_test_binary, y_pred)
        roc_auc = self._compute_auc(fpr, tpr)
            
        # 计算PR曲线数据
        precision_curve, recall_curve, _ = self._compute_pr_curve(y_test_binary, y_pred)
        pr_auc = self._compute_auc(recall_curve, precision_curve) if len(recall_curve) > 1 else 0
        
        results = {
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        }
        
        # 可视化
        if plot:
            self._plot_results(y_pred, y_test_binary, y_pred_binary, cm, fpr, tpr, roc_auc, 
                              precision_curve, recall_curve)
            
        return results
    
    def _plot_results(self, y_pred, y_test_binary, y_pred_binary, cm, fpr, tpr, roc_auc,
                     precision_curve, recall_curve):
        """
        绘制评估结果的可视化图表
        
        参数:
        各种评估指标和数据
        """
        # 创建一个3x2的图表布局
        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        
        # 1. 预测值与真实值的对比
        axes[0, 0].scatter(self.y_test, y_pred)
        axes[0, 0].plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('预测值与真实值对比')
        
        # 2. 损失曲线
        axes[0, 1].plot(range(1, len(self.history['loss'])+1), self.history['loss'])
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('MSE损失')
        axes[0, 1].set_title('训练损失曲线')
        
        # 3. 混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('预测标签')
        axes[1, 0].set_ylabel('真实标签')
        axes[1, 0].set_title('混淆矩阵')
        
        # 4. ROC曲线
        axes[1, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axes[1, 1].plot([0, 1], [0, 1], 'r--')
        axes[1, 1].set_xlabel('假正例率 (FPR)')
        axes[1, 1].set_ylabel('真正例率 (TPR)')
        axes[1, 1].set_title('ROC曲线')
        axes[1, 1].legend(loc='lower right')
        
        # 5. 精确率-召回率曲线
        axes[2, 0].plot(recall_curve, precision_curve)
        axes[2, 0].set_xlabel('召回率')
        axes[2, 0].set_ylabel('精确率')
        axes[2, 0].set_title('精确率-召回率曲线')
        
        # 6. 对于高维特征，展示特征重要性
        if self.feature_dim > 1:
            feature_importance = np.abs(self.weights)
            indices = np.argsort(feature_importance)[::-1]
            names = [f'特征 {i}' for i in range(self.feature_dim)]
            
            axes[2, 1].barh(range(self.feature_dim), feature_importance[indices])
            axes[2, 1].set_yticks(range(self.feature_dim))
            axes[2, 1].set_yticklabels([names[i] for i in indices])
            axes[2, 1].set_title('特征重要性')
        else:
            axes[2, 1].text(0.5, 0.5, '单特征模型', ha='center', va='center')
            axes[2, 1].set_title('特征重要性不适用')
        
        plt.tight_layout()
        plt.show()
    
    def get_weights(self):
        """
        获取模型的权重
        
        返回:
        模型权重和偏置
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        return {
            'weights': self.weights,
            'bias': self.bias
        }

# 使用示例函数
def demo_linear_regression():
    """
    演示如何使用线性回归模型
    """
    # 生成一些示例数据
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100个样本，3个特征
    # 生成目标变量 (线性关系加上一些噪声)
    true_weights = [0.5, -0.2, 0.7]
    true_bias = 0.3
    y = np.dot(X, true_weights) + true_bias + 0.1 * np.random.randn(100)
    
    # 实例化并训练模型
    model = LinearRegressionModel(learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)
    
    # 评估模型
    metrics = model.evaluate()
    
    # 输出评估指标
    print("均方误差 (MSE):", metrics['mse'])
    print("R² 分数:", metrics['r2'])
    print("准确率:", metrics['accuracy'])
    print("精确率:", metrics['precision'])
    print("召回率:", metrics['recall'])
    print("F1 分数:", metrics['f1'])
    print("ROC AUC:", metrics['roc_auc'])
    print("PR AUC:", metrics['pr_auc'])
    
    # 获取模型权重
    weights = model.get_weights()
    print("\n模型权重:", weights['weights'])
    print("模型偏置:", weights['bias'])
    
    # 预测新数据
    X_new = np.random.randn(5, 3)
    predictions = model.predict(X_new)
    print("\n新数据预测结果:", predictions)
