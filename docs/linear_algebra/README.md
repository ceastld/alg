# 线性代数文档

本目录包含关于线性代数概念的全面文档，重点关注矩阵理论及其在机器学习和数据科学中的应用。

## 目录

### 核心概念
- [矩阵基础](matrix_basics.md) - 基本矩阵概念、类型和性质
- [矩阵运算](matrix_operations.md) - 加法、乘法、转置和其他运算
- [特征值和特征向量](eigenvalues_eigenvectors.md) - 特征值和特征向量
- [矩阵分解](matrix_decomposition.md) - LU、QR、SVD等分解方法

### 高级主题
- [线性变换](linear_transformations.md) - 几何解释和应用
- [向量空间](vector_spaces.md) - 基、维度和子空间
- [行列式](determinants.md) - 性质和计算方法
- [矩阵范数](matrix_norms.md) - 不同类型的矩阵范数

### 应用
- [主成分分析](pca.md) - 使用特征值进行降维
- [线性回归](linear_regression_matrix.md) - 线性回归的矩阵形式
- [最小二乘法](least_squares_matrix.md) - 最小二乘问题的矩阵方法

## 数学符号

在本文档中，我们使用标准数学符号：
- **粗体大写字母** (A, B, C) 表示矩阵
- **粗体小写字母** (v, w, x) 表示向量
- **斜体小写字母** (a, b, c) 表示标量
- **上标 T** (A^T) 表示矩阵转置
- **上标 -1** (A^-1) 表示矩阵逆

## 前置知识

- 代数和微积分的基础知识
- 向量运算的熟悉程度
- Python 和 NumPy 的基础知识（用于代码示例）

## 快速参考

### 常见矩阵类型
- **方阵**: n × n 矩阵
- **单位矩阵**: I（对角线元素为1，其他为0）
- **对角矩阵**: 只有对角线元素非零
- **对称矩阵**: A = A^T
- **正交矩阵**: A^T A = I

### 关键性质
- **特征值**: det(A - λI) = 0 的解
- **迹**: 对角线元素的和
- **行列式**: 表征矩阵的标量值
- **秩**: 列空间的维度

## 代码示例

所有代码示例都使用 NumPy 进行矩阵运算：

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2], [3, 4]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 矩阵乘法
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
```

## 延伸阅读

- Gilbert Strang, "Introduction to Linear Algebra"
- David C. Lay, "Linear Algebra and Its Applications"
- Trefethen & Bau, "Numerical Linear Algebra"
