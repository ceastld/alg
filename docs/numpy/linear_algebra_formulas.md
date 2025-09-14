# 线性代数公式与 NumPy 实现

## 目录
- [矩阵基础](#矩阵基础)
- [矩阵运算](#矩阵运算)
- [行列式与逆矩阵](#行列式与逆矩阵)
- [特征值与特征向量](#特征值与特征向量)
- [线性方程组](#线性方程组)
- [向量空间](#向量空间)
- [正交化](#正交化)
- [矩阵分解](#矩阵分解)

## 矩阵基础

### 矩阵创建与基本属性
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# 创建矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[2, 0, 1], [1, 3, 2], [0, 1, 1]])

print("矩阵 A:")
print(A)
print(f"矩阵 A 的形状: {A.shape}")
print(f"矩阵 A 的秩: {np.linalg.matrix_rank(A)}")
print(f"矩阵 A 的迹: {np.trace(A)}")

# 特殊矩阵
def create_special_matrices(n):
    """创建特殊矩阵"""
    # 单位矩阵
    I = np.eye(n)
    
    # 零矩阵
    Z = np.zeros((n, n))
    
    # 全1矩阵
    O = np.ones((n, n))
    
    # 对角矩阵
    D = np.diag([1, 2, 3, 4])
    
    # 上三角矩阵
    U = np.triu(np.random.rand(n, n))
    
    # 下三角矩阵
    L = np.tril(np.random.rand(n, n))
    
    return I, Z, O, D, U, L

I, Z, O, D, U, L = create_special_matrices(4)
print("单位矩阵 I:")
print(I)
print("对角矩阵 D:")
print(D)
```

### 矩阵转置与共轭
```python
def matrix_transpose_properties(A):
    """矩阵转置的性质"""
    print("原矩阵 A:")
    print(A)
    
    # 转置
    A_T = A.T
    print("转置矩阵 A^T:")
    print(A_T)
    
    # 转置的转置等于原矩阵
    print(f"(A^T)^T = A: {np.allclose((A_T.T), A)}")
    
    # 如果矩阵是方阵，检查对称性
    if A.shape[0] == A.shape[1]:
        is_symmetric = np.allclose(A, A_T)
        print(f"矩阵是否对称: {is_symmetric}")
        
        # 反对称矩阵: A^T = -A
        is_antisymmetric = np.allclose(A_T, -A)
        print(f"矩阵是否反对称: {is_antisymmetric}")

# 示例
A = np.array([[1, 2, 3], [4, 5, 6]])
matrix_transpose_properties(A)

# 对称矩阵示例
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
matrix_transpose_properties(S)
```

## 矩阵运算

### 基本运算
```python
def matrix_operations(A, B):
    """矩阵基本运算"""
    print("矩阵 A:")
    print(A)
    print("矩阵 B:")
    print(B)
    
    # 矩阵加法
    if A.shape == B.shape:
        C_add = A + B
        print("矩阵加法 A + B:")
        print(C_add)
    
    # 矩阵减法
    if A.shape == B.shape:
        C_sub = A - B
        print("矩阵减法 A - B:")
        print(C_sub)
    
    # 标量乘法
    scalar = 3
    C_scalar = scalar * A
    print(f"标量乘法 {scalar} * A:")
    print(C_scalar)
    
    # 矩阵乘法
    if A.shape[1] == B.shape[0]:
        C_mult = np.dot(A, B)  # 或者 A @ B
        print("矩阵乘法 A @ B:")
        print(C_mult)
    
    # 元素级乘法（Hadamard积）
    if A.shape == B.shape:
        C_hadamard = A * B
        print("Hadamard积 A ⊙ B:")
        print(C_hadamard)

# 示例
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])
matrix_operations(A, B)
```

### 矩阵幂运算
```python
def matrix_power(A, n):
    """计算矩阵的n次幂"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩阵必须是方阵才能计算幂")
    
    result = np.eye(A.shape[0])  # 初始化为单位矩阵
    for _ in range(n):
        result = np.dot(result, A)
    
    return result

def matrix_power_efficient(A, n):
    """高效的矩阵幂运算（使用NumPy内置函数）"""
    return np.linalg.matrix_power(A, n)

# 示例
A = np.array([[1, 1], [1, 0]])  # 斐波那契矩阵
n = 10

power_result = matrix_power(A, n)
efficient_result = matrix_power_efficient(A, n)

print(f"矩阵 A 的 {n} 次幂:")
print(power_result)
print(f"两种方法结果相同: {np.allclose(power_result, efficient_result)}")
```

## 行列式与逆矩阵

### 行列式计算
```python
def determinant_properties(A):
    """行列式的性质"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩阵必须是方阵才能计算行列式")
    
    det_A = np.linalg.det(A)
    print(f"矩阵 A 的行列式: det(A) = {det_A:.6f}")
    
    # 行列式的性质
    # 1. det(A^T) = det(A)
    det_A_T = np.linalg.det(A.T)
    print(f"det(A^T) = {det_A_T:.6f}")
    print(f"det(A^T) = det(A): {np.isclose(det_A, det_A_T)}")
    
    # 2. det(AB) = det(A)det(B)
    if A.shape[0] == A.shape[1]:
        B = np.random.rand(A.shape[0], A.shape[1])
        det_AB = np.linalg.det(A @ B)
        det_A_det_B = np.linalg.det(A) * np.linalg.det(B)
        print(f"det(AB) = {det_AB:.6f}")
        print(f"det(A)det(B) = {det_A_det_B:.6f}")
        print(f"det(AB) = det(A)det(B): {np.isclose(det_AB, det_A_det_B)}")
    
    # 3. 如果A是上三角或下三角矩阵，行列式等于对角元素的乘积
    if np.allclose(A, np.triu(A)):  # 上三角
        diag_product = np.prod(np.diag(A))
        print(f"上三角矩阵，对角元素乘积: {diag_product:.6f}")
        print(f"与行列式相等: {np.isclose(det_A, diag_product)}")

# 示例
A = np.array([[2, 1, 0], [0, 3, 1], [0, 0, 4]])
determinant_properties(A)
```

### 逆矩阵
```python
def inverse_matrix_properties(A):
    """逆矩阵的性质"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩阵必须是方阵才能计算逆矩阵")
    
    det_A = np.linalg.det(A)
    print(f"矩阵 A 的行列式: {det_A:.6f}")
    
    if abs(det_A) < 1e-10:
        print("矩阵是奇异的，不存在逆矩阵")
        return None
    
    # 计算逆矩阵
    A_inv = np.linalg.inv(A)
    print("逆矩阵 A^(-1):")
    print(A_inv)
    
    # 验证 A * A^(-1) = I
    identity_check = A @ A_inv
    print("A * A^(-1) (应该接近单位矩阵):")
    print(identity_check)
    print(f"是否接近单位矩阵: {np.allclose(identity_check, np.eye(A.shape[0]))}")
    
    # 逆矩阵的性质
    # 1. (A^(-1))^(-1) = A
    A_inv_inv = np.linalg.inv(A_inv)
    print(f"(A^(-1))^(-1) = A: {np.allclose(A_inv_inv, A)}")
    
    # 2. (A^T)^(-1) = (A^(-1))^T
    A_T_inv = np.linalg.inv(A.T)
    A_inv_T = A_inv.T
    print(f"(A^T)^(-1) = (A^(-1))^T: {np.allclose(A_T_inv, A_inv_T)}")
    
    return A_inv

# 示例
A = np.array([[2, 1], [1, 1]])
A_inv = inverse_matrix_properties(A)
```

### 伴随矩阵
```python
def adjugate_matrix(A):
    """计算伴随矩阵（余子矩阵）"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩阵必须是方阵")
    
    n = A.shape[0]
    adj = np.zeros_like(A)
    
    for i in range(n):
        for j in range(n):
            # 计算余子式
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            cofactor = (-1)**(i+j) * np.linalg.det(minor)
            adj[j, i] = cofactor  # 注意转置
    
    return adj

def inverse_by_adjugate(A):
    """通过伴随矩阵计算逆矩阵: A^(-1) = adj(A) / det(A)"""
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-10:
        raise ValueError("矩阵是奇异的，不存在逆矩阵")
    
    adj_A = adjugate_matrix(A)
    A_inv = adj_A / det_A
    return A_inv

# 示例
A = np.array([[1, 2], [3, 4]])
A_inv_adj = inverse_by_adjugate(A)
A_inv_np = np.linalg.inv(A)

print("通过伴随矩阵计算的逆矩阵:")
print(A_inv_adj)
print("NumPy计算的逆矩阵:")
print(A_inv_np)
print(f"两种方法结果相同: {np.allclose(A_inv_adj, A_inv_np)}")
```

## 特征值与特征向量

### 特征值分解
```python
def eigenvalue_decomposition(A):
    """特征值分解"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩阵必须是方阵")
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("特征值:")
    print(eigenvalues)
    print("特征向量 (列向量):")
    print(eigenvectors)
    
    # 验证特征值方程 Av = λv
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ eigenvec
        lambda_v = eigenval * eigenvec
        print(f"特征值 {i+1}: λ = {eigenval:.6f}")
        print(f"Av = λv 验证: {np.allclose(Av, lambda_v)}")
    
    # 重构原矩阵: A = VΛV^(-1)
    V = eigenvectors
    Lambda = np.diag(eigenvalues)
    V_inv = np.linalg.inv(V)
    A_reconstructed = V @ Lambda @ V_inv
    
    print(f"重构矩阵与原矩阵相同: {np.allclose(A, A_reconstructed)}")
    
    return eigenvalues, eigenvectors

# 示例
A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = eigenvalue_decomposition(A)
```

### 特征值的性质
```python
def eigenvalue_properties(A):
    """特征值的性质"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"矩阵 A 的特征值: {eigenvalues}")
    
    # 1. 特征值的和等于矩阵的迹
    trace_A = np.trace(A)
    sum_eigenvalues = np.sum(eigenvalues)
    print(f"矩阵的迹: {trace_A}")
    print(f"特征值的和: {sum_eigenvalues}")
    print(f"迹等于特征值之和: {np.isclose(trace_A, sum_eigenvalues)}")
    
    # 2. 特征值的乘积等于矩阵的行列式
    det_A = np.linalg.det(A)
    product_eigenvalues = np.prod(eigenvalues)
    print(f"矩阵的行列式: {det_A}")
    print(f"特征值的乘积: {product_eigenvalues}")
    print(f"行列式等于特征值乘积: {np.isclose(det_A, product_eigenvalues)}")
    
    # 3. 如果A是对称矩阵，特征值是实数
    is_symmetric = np.allclose(A, A.T)
    print(f"矩阵是否对称: {is_symmetric}")
    if is_symmetric:
        print(f"对称矩阵的特征值都是实数: {np.all(np.isreal(eigenvalues))}")
    
    # 4. 特征向量的正交性（对于对称矩阵）
    if is_symmetric:
        V = eigenvectors
        V_T_V = V.T @ V
        print(f"对称矩阵的特征向量正交: {np.allclose(V_T_V, np.eye(V.shape[0]))}")

# 示例
A = np.array([[2, 1], [1, 2]])  # 对称矩阵
eigenvalue_properties(A)
```

## 线性方程组

### 高斯消元法
```python
def gaussian_elimination(A, b):
    """高斯消元法求解线性方程组 Ax = b"""
    n = A.shape[0]
    
    # 构造增广矩阵
    augmented = np.column_stack([A, b])
    
    print("增广矩阵:")
    print(augmented)
    
    # 前向消元
    for i in range(n):
        # 寻找主元
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k, i]) > abs(augmented[max_row, i]):
                max_row = k
        
        # 交换行
        augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # 消元
        for k in range(i + 1, n):
            if abs(augmented[i, i]) > 1e-10:  # 避免除零
                factor = augmented[k, i] / augmented[i, i]
                augmented[k, i:] -= factor * augmented[i, i:]
    
    print("消元后的增广矩阵:")
    print(augmented)
    
    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i, n]
        for j in range(i + 1, n):
            x[i] -= augmented[i, j] * x[j]
        if abs(augmented[i, i]) > 1e-10:
            x[i] /= augmented[i, i]
    
    return x

def solve_linear_system(A, b):
    """求解线性方程组的多种方法"""
    print("系数矩阵 A:")
    print(A)
    print("常数向量 b:")
    print(b)
    
    # 方法1: 使用NumPy的solve函数
    x_np = np.linalg.solve(A, b)
    print("NumPy求解结果:")
    print(x_np)
    
    # 方法2: 高斯消元法
    x_gauss = gaussian_elimination(A.copy(), b.copy())
    print("高斯消元法求解结果:")
    print(x_gauss)
    
    # 方法3: 使用逆矩阵
    if np.linalg.det(A) != 0:
        A_inv = np.linalg.inv(A)
        x_inv = A_inv @ b
        print("逆矩阵求解结果:")
        print(x_inv)
    
    # 验证解
    residual = A @ x_np - b
    print(f"残差: {residual}")
    print(f"解是否正确: {np.allclose(residual, 0)}")
    
    return x_np

# 示例
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)
x = solve_linear_system(A, b)
```

### 最小二乘法
```python
def least_squares(A, b):
    """最小二乘法求解超定方程组"""
    print("系数矩阵 A:")
    print(A)
    print("常数向量 b:")
    print(b)
    
    # 方法1: 正规方程 (A^T A)^(-1) A^T b
    A_T_A = A.T @ A
    A_T_b = A.T @ b
    x_normal = np.linalg.solve(A_T_A, A_T_b)
    print("正规方程求解结果:")
    print(x_normal)
    
    # 方法2: 使用NumPy的lstsq函数
    x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("NumPy lstsq求解结果:")
    print(x_lstsq)
    print(f"残差平方和: {residuals}")
    print(f"矩阵的秩: {rank}")
    
    # 计算残差
    residual_vector = A @ x_lstsq - b
    residual_norm = np.linalg.norm(residual_vector)
    print(f"残差向量的范数: {residual_norm}")
    
    return x_lstsq

# 示例：拟合直线 y = ax + b
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 3.9, 6.1, 8.0, 9.9])

# 构造系数矩阵 A = [[x1, 1], [x2, 1], ...]
A = np.column_stack([x_data, np.ones(len(x_data))])
b = y_data

x_fit = least_squares(A, b)
print(f"拟合直线: y = {x_fit[0]:.3f}x + {x_fit[1]:.3f}")
```

## 向量空间

### 向量运算
```python
def vector_operations():
    """向量基本运算"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print("向量 v1:", v1)
    print("向量 v2:", v2)
    
    # 向量加法
    v_add = v1 + v2
    print("向量加法 v1 + v2:", v_add)
    
    # 标量乘法
    scalar = 3
    v_scalar = scalar * v1
    print(f"标量乘法 {scalar} * v1:", v_scalar)
    
    # 点积（内积）
    dot_product = np.dot(v1, v2)
    print("点积 v1 · v2:", dot_product)
    
    # 向量范数
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    print(f"向量 v1 的范数: {norm_v1:.6f}")
    print(f"向量 v2 的范数: {norm_v2:.6f}")
    
    # 单位向量
    unit_v1 = v1 / norm_v1
    print("v1 的单位向量:", unit_v1)
    print(f"单位向量的范数: {np.linalg.norm(unit_v1):.6f}")
    
    # 向量夹角
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"向量夹角: {angle_deg:.2f} 度")
    
    # 叉积（仅适用于3维向量）
    if len(v1) == 3:
        cross_product = np.cross(v1, v2)
        print("叉积 v1 × v2:", cross_product)

vector_operations()
```

### 线性相关性与基
```python
def linear_independence(vectors):
    """判断向量组的线性相关性"""
    print("向量组:")
    for i, v in enumerate(vectors):
        print(f"v{i+1}: {v}")
    
    # 构造矩阵，每列是一个向量
    A = np.column_stack(vectors)
    print("向量矩阵 A:")
    print(A)
    
    # 计算矩阵的秩
    rank = np.linalg.matrix_rank(A)
    print(f"矩阵的秩: {rank}")
    print(f"向量个数: {len(vectors)}")
    
    if rank == len(vectors):
        print("向量组线性无关")
    else:
        print("向量组线性相关")
    
    # 如果线性相关，找到线性关系
    if rank < len(vectors):
        # 使用SVD分解找到零空间
        U, s, Vt = np.linalg.svd(A)
        # 零空间向量对应最小的奇异值
        null_space = Vt[-1, :]  # 最后一行
        print("线性关系系数:", null_space)
        
        # 验证线性关系
        linear_combination = np.sum([null_space[i] * vectors[i] for i in range(len(vectors))], axis=0)
        print(f"线性组合结果: {linear_combination}")
        print(f"是否为零向量: {np.allclose(linear_combination, 0)}")
    
    return rank == len(vectors)

# 示例
vectors1 = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
vectors2 = [np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([1, 1, 1])]

print("线性无关的向量组:")
linear_independence(vectors1)
print("\n线性相关的向量组:")
linear_independence(vectors2)
```

## 正交化

### Gram-Schmidt正交化
```python
def gram_schmidt(vectors):
    """Gram-Schmidt正交化过程"""
    print("原始向量组:")
    for i, v in enumerate(vectors):
        print(f"v{i+1}: {v}")
    
    orthogonal_vectors = []
    
    for i, v in enumerate(vectors):
        # 当前向量
        current = v.copy()
        
        # 减去与前面所有正交向量的投影
        for j in range(i):
            projection = np.dot(current, orthogonal_vectors[j]) / np.dot(orthogonal_vectors[j], orthogonal_vectors[j])
            current -= projection * orthogonal_vectors[j]
        
        # 如果向量不为零，则加入正交向量组
        if not np.allclose(current, 0):
            orthogonal_vectors.append(current)
            print(f"正交向量 u{i+1}: {current}")
    
    # 单位化
    orthonormal_vectors = []
    for v in orthogonal_vectors:
        unit_v = v / np.linalg.norm(v)
        orthonormal_vectors.append(unit_v)
        print(f"单位向量 e{i+1}: {unit_v}")
    
    return orthonormal_vectors

def verify_orthogonality(vectors):
    """验证向量的正交性"""
    print("验证正交性:")
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_product = np.dot(vectors[i], vectors[j])
            print(f"e{i+1} · e{j+1} = {dot_product:.10f}")
    
    print("验证单位性:")
    for i, v in enumerate(vectors):
        norm = np.linalg.norm(v)
        print(f"||e{i+1}|| = {norm:.10f}")

# 示例
vectors = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])]
orthonormal = gram_schmidt(vectors)
verify_orthogonality(orthonormal)
```

### QR分解
```python
def qr_decomposition(A):
    """QR分解: A = QR"""
    print("原矩阵 A:")
    print(A)
    
    # 使用NumPy的QR分解
    Q, R = np.linalg.qr(A)
    
    print("正交矩阵 Q:")
    print(Q)
    print("上三角矩阵 R:")
    print(R)
    
    # 验证Q是正交矩阵
    Q_T_Q = Q.T @ Q
    print("Q^T Q (应该接近单位矩阵):")
    print(Q_T_Q)
    print(f"Q是正交矩阵: {np.allclose(Q_T_Q, np.eye(Q.shape[0]))}")
    
    # 验证分解: A = QR
    A_reconstructed = Q @ R
    print("重构矩阵 A = QR:")
    print(A_reconstructed)
    print(f"分解正确: {np.allclose(A, A_reconstructed)}")
    
    return Q, R

# 示例
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
Q, R = qr_decomposition(A)
```

## 矩阵分解

### LU分解
```python
def lu_decomposition(A):
    """LU分解: A = LU"""
    print("原矩阵 A:")
    print(A)
    
    # 使用SciPy的LU分解
    from scipy.linalg import lu
    P, L, U = lu(A)
    
    print("置换矩阵 P:")
    print(P)
    print("下三角矩阵 L:")
    print(L)
    print("上三角矩阵 U:")
    print(U)
    
    # 验证分解: PA = LU
    PA = P @ A
    LU = L @ U
    print("PA:")
    print(PA)
    print("LU:")
    print(LU)
    print(f"分解正确: {np.allclose(PA, LU)}")
    
    return P, L, U

# 示例
A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=float)
P, L, U = lu_decomposition(A)
```

### 奇异值分解 (SVD)
```python
def svd_decomposition(A):
    """奇异值分解: A = UΣV^T"""
    print("原矩阵 A:")
    print(A)
    
    # 计算SVD
    U, s, Vt = np.linalg.svd(A)
    
    print("左奇异向量矩阵 U:")
    print(U)
    print("奇异值:")
    print(s)
    print("右奇异向量矩阵 V^T:")
    print(Vt)
    
    # 构造奇异值矩阵
    Sigma = np.zeros_like(A)
    Sigma[:len(s), :len(s)] = np.diag(s)
    print("奇异值矩阵 Σ:")
    print(Sigma)
    
    # 验证分解: A = UΣV^T
    A_reconstructed = U @ Sigma @ Vt
    print("重构矩阵 A = UΣV^T:")
    print(A_reconstructed)
    print(f"分解正确: {np.allclose(A, A_reconstructed)}")
    
    # 矩阵的秩
    rank = np.sum(s > 1e-10)
    print(f"矩阵的秩: {rank}")
    
    # 条件数
    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
    print(f"条件数: {condition_number}")
    
    return U, s, Vt

# 示例
A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
U, s, Vt = svd_decomposition(A)
```

### 特征值分解与对角化
```python
def matrix_diagonalization(A):
    """矩阵对角化: A = PΛP^(-1)"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩阵必须是方阵")
    
    print("原矩阵 A:")
    print(A)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("特征值:")
    print(eigenvalues)
    print("特征向量矩阵 P:")
    print(eigenvectors)
    
    # 构造对角矩阵
    Lambda = np.diag(eigenvalues)
    print("对角矩阵 Λ:")
    print(Lambda)
    
    # 计算P的逆
    P_inv = np.linalg.inv(eigenvectors)
    print("P^(-1):")
    print(P_inv)
    
    # 验证对角化: A = PΛP^(-1)
    A_reconstructed = eigenvectors @ Lambda @ P_inv
    print("重构矩阵 A = PΛP^(-1):")
    print(A_reconstructed)
    print(f"对角化正确: {np.allclose(A, A_reconstructed)}")
    
    # 矩阵的幂
    n = 3
    A_power = eigenvectors @ (Lambda ** n) @ P_inv
    A_power_direct = np.linalg.matrix_power(A, n)
    print(f"A^{n} (通过对角化):")
    print(A_power)
    print(f"A^{n} (直接计算):")
    print(A_power_direct)
    print(f"两种方法结果相同: {np.allclose(A_power, A_power_direct)}")
    
    return eigenvalues, eigenvectors

# 示例
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalues, eigenvectors = matrix_diagonalization(A)
```

这些线性代数公式和 NumPy 实现涵盖了线性代数的核心概念，从基本的矩阵运算到高级的矩阵分解技术。通过结合理论公式和实际代码实现，可以更好地理解和应用线性代数知识。
