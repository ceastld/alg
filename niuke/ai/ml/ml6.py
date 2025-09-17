from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import random 

# 设置随机种子
random.seed(42)

# 读入数据集
iris = load_iris()
X = iris.data
y = iris.target

# 对数据集进行一次随机洗牌
indices = list(range(len(X)))
random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

# 读取测试样本数量
n = int(input())

# 数据分成训练集与测试集（最后n个作为测试集）
X_test = X_shuffled[-n:]
y_test = y_shuffled[-n:]
X_train = X_shuffled[:-n]
y_train = y_shuffled[:-n]

# 训练LogisticRegression模型
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 输出结果
for i in range(len(y_pred)):
    print(f'{iris.target_names[y_pred[i]]} {np.max(y_prob[i]):.2f}')