# 最小二乘法 (Least Squares Method)

## 概述

最小二乘法是一种数学优化技术，通过最小化误差的平方和来寻找数据的最佳函数匹配。在线性回归中，它用于找到最佳拟合直线。

## 问题设定

给定数据点 $(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$，寻找直线 $y = ax + b$ 使得所有点到直线的距离平方和最小。

## 数学推导

### 1. 目标函数

对于每个数据点 $(x_i, y_i)$：
- 预测值：$\hat{y_i} = ax_i + b$
- 误差：$e_i = y_i - \hat{y_i} = y_i - (ax_i + b)$

**残差平方和**：
$$S = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - ax_i - b)^2$$

### 2. 求偏导数

对参数 $a$ 和 $b$ 分别求偏导数并令其等于0：

**对 $a$ 求偏导**：
$$\frac{\partial S}{\partial a} = \sum_{i=1}^{n} 2(y_i - ax_i - b)(-x_i) = 0$$

展开得：
$$\sum_{i=1}^{n} x_i y_i - a\sum_{i=1}^{n} x_i^2 - b\sum_{i=1}^{n} x_i = 0$$

**对 $b$ 求偏导**：
$$\frac{\partial S}{\partial b} = \sum_{i=1}^{n} 2(y_i - ax_i - b)(-1) = 0$$

展开得：
$$\sum_{i=1}^{n} y_i - a\sum_{i=1}^{n} x_i - nb = 0$$

### 3. 建立方程组

$$\begin{cases}
a\sum_{i=1}^{n} x_i^2 + b\sum_{i=1}^{n} x_i = \sum_{i=1}^{n} x_i y_i \\
a\sum_{i=1}^{n} x_i + nb = \sum_{i=1}^{n} y_i
\end{cases}$$

### 4. 求解参数

**从第二个方程解出 $b$**：
$$b = \frac{\sum_{i=1}^{n} y_i - a\sum_{i=1}^{n} x_i}{n} = \bar{y} - a\bar{x}$$

其中：
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$（x的均值）
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$（y的均值）

**将 $b$ 代入第一个方程求解 $a$**：
$$a = \frac{\sum_{i=1}^{n} x_i y_i - n\bar{x}\bar{y}}{\sum_{i=1}^{n} x_i^2 - n\bar{x}^2}$$

### 5. 最终公式

**斜率**：
$$a = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

**截距**：
$$b = \bar{y} - a\bar{x}$$

## 协方差形式

定义：
- **协方差**：$S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$
- **x的方差**：$S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$
- **y的方差**：$S_{yy} = \sum_{i=1}^{n} (y_i - \bar{y})^2$

则公式简化为：
$$a = \frac{S_{xy}}{S_{xx}}$$
$$b = \bar{y} - a\bar{x}$$

## 相关系数

**皮尔逊相关系数**：
$$r = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**相关系数的性质**：
- $-1 \leq r \leq 1$
- $r > 0$：正相关
- $r < 0$：负相关
- $|r|$ 接近1：强相关
- $|r|$ 接近0：弱相关

## 几何意义

1. **垂直距离**：最小化的是点到直线的垂直距离的平方和
2. **重心**：拟合直线必然通过数据点的重心 $(\bar{x}, \bar{y})$
3. **残差**：所有残差的和为零：$\sum_{i=1}^{n} e_i = 0$

## 矩阵形式

对于多元线性回归 $y = X\beta + \epsilon$：

**正规方程**：
$$\beta = (X^T X)^{-1} X^T y$$

其中：
- $X$ 是设计矩阵
- $y$ 是响应向量
- $\beta$ 是参数向量

## 应用场景

1. **线性回归**：预测连续值
2. **曲线拟合**：多项式回归
3. **数据平滑**：去除噪声
4. **参数估计**：系统辨识
5. **信号处理**：滤波器设计

## 优缺点

### 优点
- 数学基础扎实
- 计算简单高效
- 有解析解
- 统计性质良好

### 缺点
- 对异常值敏感
- 假设线性关系
- 要求误差正态分布
- 可能过拟合

## 扩展方法

1. **加权最小二乘**：给不同点不同权重
2. **正则化最小二乘**：Ridge回归、Lasso回归
3. **鲁棒回归**：Huber损失、M-估计
4. **非线性最小二乘**：Levenberg-Marquardt算法

## 实现示例

```python
import numpy as np

def least_squares(x, y):
    """最小二乘法拟合直线"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算协方差和方差
    S_xy = np.sum((x - x_mean) * (y - y_mean))
    S_xx = np.sum((x - x_mean) ** 2)
    
    # 计算斜率和截距
    a = S_xy / S_xx
    b = y_mean - a * x_mean
    
    return a, b

# 使用示例
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
a, b = least_squares(x, y)
print(f"拟合直线: y = {a:.2f}x + {b:.2f}")
```

## 总结

最小二乘法是统计学和机器学习中的基础方法，通过最小化残差平方和来找到最佳拟合参数。它不仅在理论上有重要意义，在实际应用中也广泛使用，是理解更复杂回归方法的重要基础。
