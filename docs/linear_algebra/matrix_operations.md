# 矩阵运算

## 矩阵乘法

### 定义

对于矩阵 A (m×n) 和矩阵 B (n×p)，它们的乘积 C = AB 是一个 m×p 矩阵，其中：

```
cᵢⱼ = Σₖ₌₁ⁿ aᵢₖbₖⱼ
```

### 基本矩阵乘法

```python
import numpy as np

def matrix_multiplication_basic():
    """演示基本矩阵乘法"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # 使用 @ 操作符
    C = A @ B
    print("矩阵乘法:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"A @ B = \n{C}")
    
    # 使用 np.dot
    D = np.dot(A, B)
    print(f"np.dot(A, B) = \n{D}")
    
    # 验证结果相同
    print(f"结果相同: {np.allclose(C, D)}")

matrix_multiplication_basic()
```

### 矩阵乘法的性质

```python
def matrix_multiplication_properties():
    """演示矩阵乘法的性质"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    
    # 结合律: (AB)C = A(BC)
    left = (A @ B) @ C
    right = A @ (B @ C)
    print(f"结合律 (AB)C = A(BC): {np.allclose(left, right)}")
    
    # 分配律: A(B + C) = AB + AC
    left = A @ (B + C)
    right = A @ B + A @ C
    print(f"分配律 A(B + C) = AB + AC: {np.allclose(left, right)}")
    
    # 标量乘法: k(AB) = (kA)B = A(kB)
    k = 3
    left = k * (A @ B)
    middle = (k * A) @ B
    right = A @ (k * B)
    print(f"标量乘法 k(AB) = (kA)B = A(kB): {np.allclose(left, middle) and np.allclose(middle, right)}")

matrix_multiplication_properties()
```

### 矩阵乘法的几何意义

```python
def matrix_multiplication_geometry():
    """演示矩阵乘法的几何意义"""
    # 旋转矩阵
    theta = np.pi / 6  # 30度
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    # 缩放矩阵
    scale = np.array([[2, 0],
                     [0, 0.5]])
    
    # 组合变换：先旋转再缩放
    combined = scale @ rotation
    
    # 测试向量
    v = np.array([1, 0])
    
    print("矩阵乘法的几何意义:")
    print(f"原始向量: {v}")
    print(f"旋转后: {rotation @ v}")
    print(f"缩放后: {scale @ v}")
    print(f"组合变换后: {combined @ v}")
    
    # 注意：矩阵乘法的顺序很重要
    different_order = rotation @ scale
    print(f"不同顺序的组合变换: {different_order @ v}")

matrix_multiplication_geometry()
```

## 矩阵的逆

### 定义

对于方阵 A，如果存在矩阵 B 使得 AB = BA = I，则称 B 是 A 的逆矩阵，记作 A⁻¹。

### 计算矩阵逆

```python
def matrix_inverse():
    """演示矩阵逆的计算"""
    # 可逆矩阵
    A = np.array([[4, 7],
                  [2, 6]])
    
    try:
        A_inv = np.linalg.inv(A)
        print("矩阵逆:")
        print(f"A = \n{A}")
        print(f"A⁻¹ = \n{A_inv}")
        
        # 验证 A * A⁻¹ = I
        identity = A @ A_inv
        print(f"A * A⁻¹ = \n{identity}")
        print(f"是否接近单位矩阵: {np.allclose(identity, np.eye(2))}")
        
    except np.linalg.LinAlgError:
        print("矩阵不可逆")
    
    # 奇异矩阵（不可逆）
    B = np.array([[1, 2],
                  [2, 4]])
    
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        print(f"\n矩阵 B = \n{B} 是奇异的（不可逆）")
        print(f"B 的行列式: {np.linalg.det(B)}")

matrix_inverse()
```

### 逆矩阵的性质

```python
def inverse_properties():
    """演示逆矩阵的性质"""
    A = np.array([[3, 1],
                  [2, 4]])
    B = np.array([[2, 1],
                  [1, 3]])
    
    A_inv = np.linalg.inv(A)
    B_inv = np.linalg.inv(B)
    
    # (A⁻¹)⁻¹ = A
    print(f"(A⁻¹)⁻¹ = A: {np.allclose(np.linalg.inv(A_inv), A)}")
    
    # (AB)⁻¹ = B⁻¹A⁻¹
    left = np.linalg.inv(A @ B)
    right = B_inv @ A_inv
    print(f"(AB)⁻¹ = B⁻¹A⁻¹: {np.allclose(left, right)}")
    
    # (A^T)⁻¹ = (A⁻¹)^T
    left = np.linalg.inv(A.T)
    right = A_inv.T
    print(f"(A^T)⁻¹ = (A⁻¹)^T: {np.allclose(left, right)}")

inverse_properties()
```

## 矩阵的转置

### 定义和计算

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

### 转置的性质

```python
def transpose_properties():
    """演示转置的性质"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # (A^T)^T = A
    print(f"(A^T)^T = A: {np.allclose((A.T).T, A)}")
    
    # (A + B)^T = A^T + B^T
    left = (A + B).T
    right = A.T + B.T
    print(f"(A + B)^T = A^T + B^T: {np.allclose(left, right)}")
    
    # (AB)^T = B^T A^T
    left = (A @ B).T
    right = B.T @ A.T
    print(f"(AB)^T = B^T A^T: {np.allclose(left, right)}")
    
    # (kA)^T = kA^T
    k = 3
    left = (k * A).T
    right = k * A.T
    print(f"(kA)^T = kA^T: {np.allclose(left, right)}")

transpose_properties()
```

## 矩阵的幂

### 矩阵的整数幂

```python
def matrix_power():
    """演示矩阵的幂运算"""
    A = np.array([[1, 1],
                  [0, 1]])
    
    print("矩阵的幂:")
    print(f"A = \n{A}")
    
    # 计算 A², A³, A⁴
    for n in range(2, 5):
        A_n = np.linalg.matrix_power(A, n)
        print(f"A^{n} = \n{A_n}")
    
    # 使用 @ 操作符计算小幂次
    A_squared = A @ A
    print(f"A² (使用 @): \n{A_squared}")

matrix_power()
```

### 矩阵幂的性质

```python
def matrix_power_properties():
    """演示矩阵幂的性质"""
    A = np.array([[2, 1],
                  [0, 3]])
    
    # A^m * A^n = A^(m+n)
    m, n = 2, 3
    left = np.linalg.matrix_power(A, m) @ np.linalg.matrix_power(A, n)
    right = np.linalg.matrix_power(A, m + n)
    print(f"A^m * A^n = A^(m+n): {np.allclose(left, right)}")
    
    # (A^m)^n = A^(mn)
    left = np.linalg.matrix_power(np.linalg.matrix_power(A, m), n)
    right = np.linalg.matrix_power(A, m * n)
    print(f"(A^m)^n = A^(mn): {np.allclose(left, right)}")

matrix_power_properties()
```

## 矩阵的范数

### 各种矩阵范数

```python
def matrix_norms():
    """演示各种矩阵范数"""
    A = np.array([[1, -2, 3],
                  [4, 5, -6],
                  [-7, 8, 9]])
    
    print("矩阵范数:")
    print(f"A = \n{A}")
    
    # Frobenius 范数（F-范数）
    frobenius_norm = np.linalg.norm(A, 'fro')
    print(f"Frobenius 范数: {frobenius_norm}")
    
    # 1-范数（列和的最大值）
    norm_1 = np.linalg.norm(A, 1)
    print(f"1-范数: {norm_1}")
    
    # 无穷范数（行和的最大值）
    norm_inf = np.linalg.norm(A, np.inf)
    print(f"无穷范数: {norm_inf}")
    
    # 2-范数（最大奇异值）
    norm_2 = np.linalg.norm(A, 2)
    print(f"2-范数: {norm_2}")

matrix_norms()
```

## 矩阵的迹

### 迹的定义和性质

```python
def matrix_trace():
    """演示矩阵迹的计算和性质"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    B = np.array([[2, 1, 0],
                  [1, 3, 2],
                  [0, 2, 4]])
    
    print("矩阵的迹:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    
    # 计算迹
    trace_A = np.trace(A)
    trace_B = np.trace(B)
    print(f"tr(A) = {trace_A}")
    print(f"tr(B) = {trace_B}")
    
    # 迹的性质
    # tr(A + B) = tr(A) + tr(B)
    print(f"tr(A + B) = tr(A) + tr(B): {np.trace(A + B) == trace_A + trace_B}")
    
    # tr(AB) = tr(BA)
    print(f"tr(AB) = tr(BA): {np.trace(A @ B) == np.trace(B @ A)}")
    
    # tr(A^T) = tr(A)
    print(f"tr(A^T) = tr(A): {np.trace(A.T) == trace_A}")

matrix_trace()
```

## 矩阵的行列式

### 行列式的计算

```python
def matrix_determinant():
    """演示矩阵行列式的计算"""
    # 2×2 矩阵
    A = np.array([[3, 1],
                  [2, 4]])
    
    det_A = np.linalg.det(A)
    print("矩阵行列式:")
    print(f"A = \n{A}")
    print(f"det(A) = {det_A}")
    
    # 3×3 矩阵
    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    det_B = np.linalg.det(B)
    print(f"\nB = \n{B}")
    print(f"det(B) = {det_B}")
    
    # 行列式的几何意义
    print(f"\n行列式的几何意义:")
    print(f"|det(A)| 表示线性变换 A 对面积的缩放因子")

matrix_determinant()
```

### 行列式的性质

```python
def determinant_properties():
    """演示行列式的性质"""
    A = np.array([[2, 1],
                  [3, 4]])
    B = np.array([[1, 2],
                  [0, 3]])
    
    # det(AB) = det(A)det(B)
    left = np.linalg.det(A @ B)
    right = np.linalg.det(A) * np.linalg.det(B)
    print(f"det(AB) = det(A)det(B): {np.isclose(left, right)}")
    
    # det(A^T) = det(A)
    print(f"det(A^T) = det(A): {np.isclose(np.linalg.det(A.T), np.linalg.det(A))}")
    
    # det(A⁻¹) = 1/det(A)
    A_inv = np.linalg.inv(A)
    print(f"det(A⁻¹) = 1/det(A): {np.isclose(np.linalg.det(A_inv), 1/np.linalg.det(A))}")
    
    # det(kA) = k^n det(A)
    k = 3
    n = A.shape[0]
    left = np.linalg.det(k * A)
    right = k**n * np.linalg.det(A)
    print(f"det(kA) = k^n det(A): {np.isclose(left, right)}")

determinant_properties()
```

## 矩阵运算的数值稳定性

### 条件数的影响

```python
def numerical_stability():
    """演示矩阵运算的数值稳定性"""
    # 良条件矩阵
    A_good = np.array([[1, 0],
                       [0, 1]])
    
    # 病条件矩阵
    A_bad = np.array([[1, 1],
                      [1, 1.0001]])
    
    print("数值稳定性:")
    print(f"良条件矩阵的条件数: {np.linalg.cond(A_good):.2e}")
    print(f"病条件矩阵的条件数: {np.linalg.cond(A_bad):.2e}")
    
    # 计算逆矩阵的精度
    try:
        A_good_inv = np.linalg.inv(A_good)
        A_bad_inv = np.linalg.inv(A_bad)
        
        # 验证 A * A⁻¹ ≈ I
        error_good = np.linalg.norm(A_good @ A_good_inv - np.eye(2))
        error_bad = np.linalg.norm(A_bad @ A_bad_inv - np.eye(2))
        
        print(f"良条件矩阵的逆矩阵误差: {error_good:.2e}")
        print(f"病条件矩阵的逆矩阵误差: {error_bad:.2e}")
        
    except np.linalg.LinAlgError:
        print("病条件矩阵可能无法求逆")

numerical_stability()
```

## 总结

矩阵运算是线性代数的核心，包括：

1. **矩阵乘法**: 最重要的运算，具有结合律和分配律
2. **矩阵逆**: 用于求解线性方程组
3. **矩阵转置**: 改变矩阵的行列结构
4. **矩阵幂**: 重复应用线性变换
5. **矩阵范数**: 衡量矩阵的"大小"
6. **矩阵迹**: 对角线元素的和
7. **矩阵行列式**: 衡量线性变换的缩放因子

理解这些运算的性质和数值特性对于正确使用矩阵进行科学计算至关重要。
