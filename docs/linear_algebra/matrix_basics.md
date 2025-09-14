# 矩阵基础

## 定义

矩阵是一个由数字排列成的矩形数组，通常用大写字母表示。一个 m×n 矩阵有 m 行 n 列。

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [ ...  ...  ...  ...]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

## 基本概念

### 矩阵元素

- **aᵢⱼ**: 矩阵 A 的第 i 行第 j 列元素
- **行向量**: 只有一行的矩阵
- **列向量**: 只有一列的矩阵

### 矩阵维度

- **m×n 矩阵**: m 行 n 列
- **方阵**: m = n 的矩阵
- **向量**: 1×n 或 m×1 的矩阵

## 特殊矩阵类型

### 单位矩阵

对角线元素为 1，其他元素为 0 的方阵：

```python
import numpy as np

# 3×3 单位矩阵
I = np.eye(3)
print(I)
# 输出:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 零矩阵

所有元素都为 0 的矩阵：

```python
# 2×3 零矩阵
Z = np.zeros((2, 3))
print(Z)
# 输出:
# [[0. 0. 0.]
#  [0. 0. 0.]]
```

### 对角矩阵

只有对角线元素非零的矩阵：

```python
# 创建对角矩阵
diag_elements = [1, 2, 3]
D = np.diag(diag_elements)
print(D)
# 输出:
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

### 对称矩阵

满足 A = A^T 的矩阵：

```python
# 对称矩阵示例
A = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print("A 是对称矩阵:", np.allclose(A, A.T))
```

### 反对称矩阵

满足 A = -A^T 的矩阵：

```python
# 反对称矩阵示例
A = np.array([[0, 2, -3],
              [-2, 0, 4],
              [3, -4, 0]])
print("A 是反对称矩阵:", np.allclose(A, -A.T))
```

## 矩阵性质

### 矩阵的秩

矩阵的秩是线性无关的行（或列）的最大数量：

```python
def matrix_rank_example():
    """演示矩阵秩的计算"""
    # 满秩矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    rank_A = np.linalg.matrix_rank(A)
    print(f"矩阵 A 的秩: {rank_A}")
    
    # 秩为 2 的矩阵
    B = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    rank_B = np.linalg.matrix_rank(B)
    print(f"矩阵 B 的秩: {rank_B}")

matrix_rank_example()
```

### 矩阵的迹

方阵的迹是对角线元素的和：

```python
def matrix_trace_example():
    """演示矩阵迹的计算"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    trace_A = np.trace(A)
    print(f"矩阵 A 的迹: {trace_A}")
    
    # 迹等于特征值的和
    eigenvalues = np.linalg.eig(A)[0]
    print(f"特征值的和: {np.sum(eigenvalues)}")
    print(f"迹等于特征值和: {np.isclose(trace_A, np.sum(eigenvalues))}")

matrix_trace_example()
```

### 矩阵的行列式

方阵的行列式是一个标量值：

```python
def matrix_determinant_example():
    """演示矩阵行列式的计算"""
    # 2×2 矩阵
    A = np.array([[3, 1],
                  [2, 4]])
    
    det_A = np.linalg.det(A)
    print(f"矩阵 A 的行列式: {det_A}")
    
    # 3×3 矩阵
    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    det_B = np.linalg.det(B)
    print(f"矩阵 B 的行列式: {det_B}")
    
    # 行列式等于特征值的乘积
    eigenvalues = np.linalg.eig(B)[0]
    print(f"特征值的乘积: {np.prod(eigenvalues)}")
    print(f"行列式等于特征值乘积: {np.isclose(det_B, np.prod(eigenvalues))}")

matrix_determinant_example()
```

## 矩阵的创建和操作

### 创建矩阵

```python
def create_matrices():
    """演示各种创建矩阵的方法"""
    # 从列表创建
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print("从列表创建矩阵:")
    print(A)
    
    # 随机矩阵
    B = np.random.rand(3, 3)
    print("\n随机矩阵:")
    print(B)
    
    # 全1矩阵
    C = np.ones((2, 3))
    print("\n全1矩阵:")
    print(C)
    
    # 全0矩阵
    D = np.zeros((3, 2))
    print("\n全0矩阵:")
    print(D)
    
    # 单位矩阵
    E = np.eye(3)
    print("\n单位矩阵:")
    print(E)

create_matrices()
```

### 矩阵索引和切片

```python
def matrix_indexing():
    """演示矩阵索引和切片操作"""
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    
    print("原始矩阵:")
    print(A)
    
    # 访问单个元素
    print(f"\nA[1, 2] = {A[1, 2]}")
    
    # 访问行
    print(f"\n第一行: {A[0, :]}")
    print(f"第二行: {A[1, :]}")
    
    # 访问列
    print(f"\n第一列: {A[:, 0]}")
    print(f"第三列: {A[:, 2]}")
    
    # 子矩阵
    print(f"\n子矩阵 A[0:2, 1:3]:")
    print(A[0:2, 1:3])

matrix_indexing()
```

## 矩阵的基本运算

### 矩阵加法

```python
def matrix_addition():
    """演示矩阵加法"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    C = A + B
    print("矩阵加法:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"A + B = \n{C}")

matrix_addition()
```

### 标量乘法

```python
def scalar_multiplication():
    """演示标量乘法"""
    A = np.array([[1, 2],
                  [3, 4]])
    scalar = 3
    
    B = scalar * A
    print("标量乘法:")
    print(f"A = \n{A}")
    print(f"{scalar} * A = \n{B}")

scalar_multiplication()
```

### 矩阵转置

```python
def matrix_transpose():
    """演示矩阵转置"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    A_T = A.T
    print("矩阵转置:")
    print(f"A = \n{A}")
    print(f"A^T = \n{A_T}")
    print(f"A 的形状: {A.shape}")
    print(f"A^T 的形状: {A_T.shape}")

matrix_transpose()
```

## 矩阵的几何意义

### 线性变换

矩阵可以表示线性变换：

```python
def linear_transformation_demo():
    """演示矩阵作为线性变换"""
    # 旋转矩阵（逆时针旋转45度）
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
    
    # 原始向量
    v = np.array([1, 0])
    
    # 变换后的向量
    v_rotated = rotation_matrix @ v
    
    print("线性变换演示:")
    print(f"旋转矩阵: \n{rotation_matrix}")
    print(f"原始向量: {v}")
    print(f"旋转后向量: {v_rotated}")
    
    # 缩放变换
    scale_matrix = np.array([[2, 0],
                            [0, 0.5]])
    
    v_scaled = scale_matrix @ v
    print(f"\n缩放矩阵: \n{scale_matrix}")
    print(f"缩放后向量: {v_scaled}")

linear_transformation_demo()
```

## 矩阵的条件数

条件数衡量矩阵的数值稳定性：

```python
def matrix_condition_number():
    """演示矩阵条件数的计算"""
    # 良条件矩阵
    A = np.array([[1, 0],
                  [0, 1]])
    cond_A = np.linalg.cond(A)
    print(f"单位矩阵的条件数: {cond_A}")
    
    # 病条件矩阵
    B = np.array([[1, 1],
                  [1, 1.0001]])
    cond_B = np.linalg.cond(B)
    print(f"病条件矩阵的条件数: {cond_B}")
    
    # 条件数的意义
    print(f"\n条件数越大，矩阵越接近奇异（不可逆）")
    print(f"条件数接近1表示矩阵数值稳定")

matrix_condition_number()
```

## 总结

矩阵是线性代数的核心概念，具有以下重要特性：

1. **基本结构**: 由数字排列成的矩形数组
2. **特殊类型**: 单位矩阵、零矩阵、对角矩阵、对称矩阵等
3. **重要性质**: 秩、迹、行列式、条件数
4. **几何意义**: 表示线性变换
5. **数值特性**: 影响计算的稳定性和精度

理解这些基础概念是学习更高级线性代数主题的前提。
