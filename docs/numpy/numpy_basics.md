# NumPy 基础方法详解

## 目录
- [数组创建](#数组创建)
- [数组操作](#数组操作)
- [数学运算](#数学运算)
- [统计函数](#统计函数)
- [线性代数](#线性代数)
- [随机数生成](#随机数生成)

## 数组创建

### 基本数组创建
```python
import numpy as np

# 创建一维数组
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)  # [1 2 3 4 5]

# 创建二维数组
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# 创建零数组
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 创建单位数组
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# 创建单位矩阵
identity = np.eye(3)
print(identity)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 特殊数组创建
```python
# 等差数列
linspace = np.linspace(0, 10, 5)  # 从0到10，生成5个等间距的数
print(linspace)  # [ 0.   2.5  5.   7.5 10. ]

# 等差数列（整数）
arange = np.arange(0, 10, 2)  # 从0到10，步长为2
print(arange)  # [0 2 4 6 8]

# 随机数组
random_arr = np.random.random((2, 3))  # 0-1之间的随机数
print(random_arr)
# [[0.12345678 0.23456789 0.34567890]
#  [0.45678901 0.56789012 0.67890123]]

# 正态分布随机数
normal_arr = np.random.normal(0, 1, (2, 3))  # 均值0，标准差1
print(normal_arr)
```

## 数组操作

### 形状操作
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 获取形状
print(arr.shape)  # (2, 3)

# 改变形状
reshaped = arr.reshape(3, 2)
print(reshaped)
# [[1 2]
#  [3 4]
#  [5 6]]

# 展平数组
flattened = arr.flatten()
print(flattened)  # [1 2 3 4 5 6]

# 转置
transposed = arr.T
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### 索引和切片
```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 基本索引
print(arr[0, 1])  # 2

# 切片
print(arr[0:2, 1:3])  # 前两行，第2-3列
# [[2 3]
#  [6 7]]

# 布尔索引
mask = arr > 5
print(mask)
# [[False False False False]
#  [False  True  True  True]
#  [ True  True  True  True]]

filtered = arr[arr > 5]
print(filtered)  # [ 6  7  8  9 10 11 12]
```

### 数组拼接
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 垂直拼接
vstack = np.vstack((arr1, arr2))
print(vstack)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 水平拼接
hstack = np.hstack((arr1, arr2))
print(hstack)
# [[1 2 5 6]
#  [3 4 7 8]]

# 使用concatenate
concat_axis0 = np.concatenate((arr1, arr2), axis=0)  # 沿axis=0拼接
concat_axis1 = np.concatenate((arr1, arr2), axis=1)  # 沿axis=1拼接
```

## 数学运算

### 基本运算
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 元素级运算
print(a + b)  # [ 6  8 10 12]
print(a - b)  # [-4 -4 -4 -4]
print(a * b)  # [ 5 12 21 32]
print(a / b)  # [0.2 0.33333333 0.42857143 0.5]

# 幂运算
print(a ** 2)  # [ 1  4  9 16]

# 三角函数
angles = np.array([0, np.pi/4, np.pi/2])
print(np.sin(angles))  # [0.         0.70710678 1.        ]
print(np.cos(angles))  # [1.         0.70710678 0.        ]
```

### 矩阵运算
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
matrix_mult = np.dot(A, B)  # 或者 A @ B
print(matrix_mult)
# [[19 22]
#  [43 50]]

# 元素级乘法
element_mult = A * B
print(element_mult)
# [[ 5 12]
#  [21 32]]

# 内积
inner_product = np.inner([1, 2, 3], [4, 5, 6])
print(inner_product)  # 32 (1*4 + 2*5 + 3*6)
```

## 统计函数

### 基本统计
```python
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 求和
print(np.sum(data))  # 45
print(np.sum(data, axis=0))  # [12 15 18] (按列求和)
print(np.sum(data, axis=1))  # [ 6 15 24] (按行求和)

# 均值
print(np.mean(data))  # 5.0
print(np.mean(data, axis=0))  # [4. 5. 6.]

# 标准差
print(np.std(data))  # 2.581988897471611

# 方差
print(np.var(data))  # 6.666666666666667

# 最大值和最小值
print(np.max(data))  # 9
print(np.min(data))  # 1
print(np.argmax(data))  # 8 (最大值的索引)
print(np.argmin(data))  # 0 (最小值的索引)
```

### 排序
```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 排序
sorted_arr = np.sort(arr)
print(sorted_arr)  # [1 1 2 3 4 5 6 9]

# 获取排序索引
indices = np.argsort(arr)
print(indices)  # [1 3 6 0 2 4 7 5]

# 使用索引重新排列
reordered = arr[indices]
print(reordered)  # [1 1 2 3 4 5 6 9]
```

## 线性代数

### 矩阵运算
```python
A = np.array([[1, 2], [3, 4]])

# 矩阵转置
A_T = A.T
print(A_T)
# [[1 3]
#  [2 4]]

# 矩阵逆
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 验证逆矩阵
identity_check = A @ A_inv
print(identity_check)
# [[1.00000000e+00 0.00000000e+00]
#  [8.88178420e-16 1.00000000e+00]]

# 行列式
det = np.linalg.det(A)
print(det)  # -2.0000000000000004

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)  # [-0.37228132  5.37228132]
print("特征向量:\n", eigenvectors)
# [[-0.82456484 -0.41597356]
#  [ 0.56576746 -0.90937671]]
```

### 线性方程组求解
```python
# 求解 Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# 使用solve求解
x = np.linalg.solve(A, b)
print(x)  # [2. 3.]

# 验证解
verification = A @ x
print(verification)  # [9. 8.]
```

## 随机数生成

### 基本随机数
```python
# 设置随机种子（保证结果可重现）
np.random.seed(42)

# 均匀分布
uniform = np.random.uniform(0, 1, 5)
print(uniform)  # [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]

# 正态分布
normal = np.random.normal(0, 1, 5)
print(normal)  # [-0.1382643   0.64768854  1.52302986 -0.23415337 -0.23413696]

# 整数随机数
integers = np.random.randint(0, 10, 5)
print(integers)  # [6 3 7 4 6]

# 随机选择
choices = np.random.choice([1, 2, 3, 4, 5], size=3, replace=True)
print(choices)  # [4 1 5]
```

### 概率分布
```python
# 二项分布
binomial = np.random.binomial(10, 0.5, 5)  # n=10, p=0.5
print(binomial)  # [4 7 5 6 4]

# 泊松分布
poisson = np.random.poisson(3, 5)  # λ=3
print(poisson)  # [2 3 1 4 2]

# 指数分布
exponential = np.random.exponential(1, 5)  # λ=1
print(exponential)  # [0.13949386 0.92187429 0.43194502 0.68864197 0.41919451]
```

## 实用技巧

### 广播（Broadcasting）
```python
# 标量与数组运算
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10  # 每个元素都加10
print(result)
# [[11 12 13]
#  [14 15 16]]

# 不同形状数组运算
row = np.array([1, 2, 3])
col = np.array([[1], [2]])
print(row + col)  # 广播运算
# [[2 3 4]
#  [3 4 5]]
```

### 条件运算
```python
arr = np.array([1, 2, 3, 4, 5])

# where函数
result = np.where(arr > 3, arr, 0)  # 大于3的保持原值，否则为0
print(result)  # [0 0 0 4 5]

# 逻辑运算
mask = (arr > 2) & (arr < 5)
print(mask)  # [False False  True  True False]
print(arr[mask])  # [3 4]
```

### 数组去重
```python
arr = np.array([1, 2, 2, 3, 3, 3, 4])

# 去重
unique = np.unique(arr)
print(unique)  # [1 2 3 4]

# 去重并返回索引
unique, indices = np.unique(arr, return_index=True)
print(unique)  # [1 2 3 4]
print(indices)  # [0 1 3 6]
```

这些是 NumPy 中最常用的方法和技巧。掌握这些基础操作后，你就可以高效地进行数值计算和数据处理了。
