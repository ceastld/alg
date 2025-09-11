# 逻辑回归完整数学推导

## 1. 逻辑回归模型

### 1.1 基本概念
逻辑回归是一种用于二分类问题的线性分类算法。给定输入特征 $x \in \mathbb{R}^d$，逻辑回归模型预测输出 $y \in \{0, 1\}$。

### 1.2 Sigmoid函数
逻辑回归使用sigmoid函数将线性组合映射到概率空间：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

其中 $z = w^T x + b$，$w$ 是权重向量，$b$ 是偏置项。

**Sigmoid函数的性质：**
- 输出范围：$(0, 1)$
- 单调递增
- 在 $z = 0$ 处值为 $0.5$
- 导数：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$

## 2. 损失函数推导

### 2.1 概率模型
对于二分类问题，我们定义：
- $P(y = 1 | x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$
- $P(y = 0 | x) = 1 - P(y = 1 | x) = \frac{e^{-(w^T x + b)}}{1 + e^{-(w^T x + b)}}$

### 2.2 似然函数
对于训练集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$，似然函数为：

$$L(w, b) = \prod_{i=1}^m P(y^{(i)} | x^{(i)})$$

$$= \prod_{i=1}^m [P(y = 1 | x^{(i)})]^{y^{(i)}} [P(y = 0 | x^{(i)})]^{1-y^{(i)}}$$

$$= \prod_{i=1}^m \left[\frac{1}{1 + e^{-(w^T x^{(i)} + b)}}\right]^{y^{(i)}} \left[\frac{e^{-(w^T x^{(i)} + b)}}{1 + e^{-(w^T x^{(i)} + b)}}\right]^{1-y^{(i)}}$$

### 2.3 对数似然函数
为了简化计算，我们取对数：

$$\ell(w, b) = \log L(w, b)$$

$$= \sum_{i=1}^m \left[ y^{(i)} \log \frac{1}{1 + e^{-(w^T x^{(i)} + b)}} + (1-y^{(i)}) \log \frac{e^{-(w^T x^{(i)} + b)}}{1 + e^{-(w^T x^{(i)} + b)}} \right]$$

$$= \sum_{i=1}^m \left[ y^{(i)} \log \sigma(w^T x^{(i)} + b) + (1-y^{(i)}) \log (1 - \sigma(w^T x^{(i)} + b)) \right]$$

### 2.4 交叉熵损失函数
为了最大化似然函数，我们最小化负对数似然（交叉熵损失）：

$$J(w, b) = -\frac{1}{m} \ell(w, b)$$

$$= -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log \sigma(w^T x^{(i)} + b) + (1-y^{(i)}) \log (1 - \sigma(w^T x^{(i)} + b)) \right]$$

$$= -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log h_w(x^{(i)}) + (1-y^{(i)}) \log (1 - h_w(x^{(i)})) \right]$$

其中 $h_w(x) = \sigma(w^T x + b)$ 是假设函数。

## 3. 梯度推导

### 3.1 对权重的梯度
我们需要计算 $\frac{\partial J}{\partial w_j}$：

$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} \sum_{i=1}^m \frac{\partial}{\partial w_j} \left[ y^{(i)} \log h_w(x^{(i)}) + (1-y^{(i)}) \log (1 - h_w(x^{(i)})) \right]$$

设 $z^{(i)} = w^T x^{(i)} + b$，则 $h_w(x^{(i)}) = \sigma(z^{(i)})$。

$$\frac{\partial}{\partial w_j} \log h_w(x^{(i)}) = \frac{1}{h_w(x^{(i)})} \cdot \frac{\partial h_w(x^{(i)})}{\partial w_j}$$

$$= \frac{1}{\sigma(z^{(i)})} \cdot \sigma'(z^{(i)}) \cdot \frac{\partial z^{(i)}}{\partial w_j}$$

$$= \frac{1}{\sigma(z^{(i)})} \cdot \sigma(z^{(i)})(1 - \sigma(z^{(i)})) \cdot x_j^{(i)}$$

$$= (1 - \sigma(z^{(i)})) \cdot x_j^{(i)} = (1 - h_w(x^{(i)})) \cdot x_j^{(i)}$$

$$\frac{\partial}{\partial w_j} \log (1 - h_w(x^{(i)})) = \frac{1}{1 - h_w(x^{(i)})} \cdot \frac{\partial (1 - h_w(x^{(i)}))}{\partial w_j}$$

$$= \frac{1}{1 - h_w(x^{(i)})} \cdot (-\sigma'(z^{(i)})) \cdot x_j^{(i)}$$

$$= \frac{1}{1 - h_w(x^{(i)})} \cdot (-\sigma(z^{(i)})(1 - \sigma(z^{(i)}))) \cdot x_j^{(i)}$$

$$= -\sigma(z^{(i)}) \cdot x_j^{(i)} = -h_w(x^{(i)}) \cdot x_j^{(i)}$$

因此：

$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} (1 - h_w(x^{(i)})) x_j^{(i)} + (1-y^{(i)}) (-h_w(x^{(i)})) x_j^{(i)} \right]$$

$$= -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} (1 - h_w(x^{(i)})) - (1-y^{(i)}) h_w(x^{(i)}) \right] x_j^{(i)}$$

$$= -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} - y^{(i)} h_w(x^{(i)}) - h_w(x^{(i)}) + y^{(i)} h_w(x^{(i)}) \right] x_j^{(i)}$$

$$= -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} - h_w(x^{(i)}) \right] x_j^{(i)}$$

$$= \frac{1}{m} \sum_{i=1}^m \left[ h_w(x^{(i)}) - y^{(i)} \right] x_j^{(i)}$$

### 3.2 对偏置的梯度
类似地，计算 $\frac{\partial J}{\partial b}$：

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left[ h_w(x^{(i)}) - y^{(i)} \right]$$

### 3.3 向量化梯度
将所有梯度组合成向量形式：

$$\nabla_w J = \frac{1}{m} X^T (h_w(X) - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)})$$

其中：
- $X$ 是 $m \times d$ 的设计矩阵
- $y$ 是 $m \times 1$ 的标签向量
- $h_w(X) = \sigma(Xw + b)$ 是 $m \times 1$ 的预测向量

## 4. 梯度下降算法

### 4.1 批量梯度下降
更新规则：

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

其中 $\alpha$ 是学习率。

向量化形式：

$$w := w - \alpha \nabla_w J = w - \frac{\alpha}{m} X^T (h_w(X) - y)$$

$$b := b - \alpha \frac{\partial J}{\partial b} = b - \frac{\alpha}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)})$$

### 4.2 随机梯度下降
对于每个样本 $(x^{(i)}, y^{(i)})$：

$$w := w - \alpha (h_w(x^{(i)}) - y^{(i)}) x^{(i)}$$

$$b := b - \alpha (h_w(x^{(i)}) - y^{(i)})$$

### 4.3 小批量梯度下降
对于小批量 $B$：

$$w := w - \frac{\alpha}{|B|} \sum_{i \in B} (h_w(x^{(i)}) - y^{(i)}) x^{(i)}$$

$$b := b - \frac{\alpha}{|B|} \sum_{i \in B} (h_w(x^{(i)}) - y^{(i)})$$

## 5. 算法实现步骤

### 5.1 初始化
1. 初始化权重 $w$ 和偏置 $b$（通常设为0或小的随机值）
2. 设置学习率 $\alpha$ 和迭代次数

### 5.2 前向传播
1. 计算线性组合：$z = Xw + b$
2. 应用sigmoid函数：$h = \sigma(z)$

### 5.3 计算损失
$$J = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log h^{(i)} + (1-y^{(i)}) \log (1-h^{(i)})]$$

### 5.4 反向传播
1. 计算梯度：
   - $\nabla_w J = \frac{1}{m} X^T (h - y)$
   - $\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (h^{(i)} - y^{(i)})$

2. 更新参数：
   - $w := w - \alpha \nabla_w J$
   - $b := b - \alpha \frac{\partial J}{\partial b}$

### 5.5 重复
重复步骤2-4直到收敛或达到最大迭代次数。

## 6. 正则化

### 6.1 L2正则化
添加L2正则化项防止过拟合：

$$J_{reg}(w, b) = J(w, b) + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$

梯度更新：

$$\nabla_w J_{reg} = \nabla_w J + \frac{\lambda}{m} w$$

$$w := w - \alpha \left( \nabla_w J + \frac{\lambda}{m} w \right)$$

### 6.2 L1正则化
添加L1正则化项：

$$J_{reg}(w, b) = J(w, b) + \frac{\lambda}{m} \sum_{j=1}^d |w_j|$$

## 7. 总结

逻辑回归通过以下步骤实现二分类：

1. **模型定义**：使用sigmoid函数将线性组合映射到概率
2. **损失函数**：使用交叉熵损失函数衡量预测与真实标签的差异
3. **优化**：通过梯度下降算法最小化损失函数
4. **预测**：输出概率，通过阈值（通常为0.5）进行分类

逻辑回归具有良好的数学理论基础，计算效率高，且易于实现和理解，是机器学习中的重要算法。
