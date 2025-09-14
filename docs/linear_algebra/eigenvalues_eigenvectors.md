# 特征值和特征向量

## 定义

对于方阵 **A**，**特征值** λ 和其对应的**特征向量** **v** 满足方程：

**Av = λv**

其中 **v ≠ 0**。

## 关键概念

### 特征方程

矩阵 **A** 的特征值是特征多项式的根：

```
det(A - λI) = 0
```

其中 **I** 是单位矩阵。

### 特征空间

对于每个特征值 λ，特征空间 E_λ 是对应于 λ 的所有特征向量的集合，加上零向量：

```
E_λ = {v : Av = λv}
```

## Properties

### Basic Properties

1. **Sum of eigenvalues** = trace of A
2. **Product of eigenvalues** = determinant of A
3. If A is symmetric, all eigenvalues are real
4. If A is positive definite, all eigenvalues are positive

### Algebraic and Geometric Multiplicity

- **Algebraic multiplicity**: The multiplicity of λ as a root of the characteristic polynomial
- **Geometric multiplicity**: The dimension of the eigenspace E_λ

## Computational Methods

### Power Method

Iterative method to find the dominant eigenvalue:

```python
import numpy as np

def power_method(A, max_iter=100, tol=1e-6):
    """
    Find the dominant eigenvalue and eigenvector using power method.
    
    Args:
        A: Square matrix
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        eigenvalue, eigenvector
    """
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for i in range(max_iter):
        Av = np.dot(A, v)
        eigenvalue = np.dot(v, Av)
        v_new = Av / np.linalg.norm(Av)
        
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    
    return eigenvalue, v

# Example usage
A = np.array([[4, 1], [2, 3]])
eigenvalue, eigenvector = power_method(A)
print(f"Dominant eigenvalue: {eigenvalue}")
print(f"Corresponding eigenvector: {eigenvector}")
```

### QR Algorithm

For finding all eigenvalues:

```python
def qr_algorithm(A, max_iter=100, tol=1e-6):
    """
    Find all eigenvalues using QR algorithm.
    
    Args:
        A: Square matrix
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        eigenvalues (approximate)
    """
    A_k = A.copy()
    
    for k in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_k = np.dot(R, Q)
        
        # Check convergence (off-diagonal elements)
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
    
    return np.diag(A_k)

# Example usage
A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]])
eigenvalues = qr_algorithm(A)
print(f"Eigenvalues: {eigenvalues}")
```

## Applications

### Principal Component Analysis (PCA)

Eigenvalues and eigenvectors are fundamental to PCA:

```python
def pca_eigen(X):
    """
    Perform PCA using eigenvalue decomposition.
    
    Args:
        X: Data matrix (samples × features)
    
    Returns:
        eigenvalues, eigenvectors, explained_variance_ratio
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    return eigenvalues, eigenvectors, explained_variance_ratio

# Example usage
np.random.seed(42)
X = np.random.randn(100, 3)
eigenvalues, eigenvectors, explained_var = pca_eigen(X)
print(f"Explained variance ratios: {explained_var}")
```

### Spectral Clustering

Using eigenvalues for clustering:

```python
def spectral_clustering(W, k):
    """
    Perform spectral clustering using graph Laplacian.
    
    Args:
        W: Weight matrix (similarity matrix)
        k: Number of clusters
    
    Returns:
        cluster_labels
    """
    # Compute degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Compute Laplacian matrix
    L = D - W
    
    # Find k smallest eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = np.argsort(eigenvalues)[:k]
    U = eigenvectors[:, idx]
    
    # Normalize rows
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
    
    # Apply k-means to the normalized eigenvectors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(U_norm)
    
    return cluster_labels
```

## Special Cases

### Symmetric Matrices

For symmetric matrices, eigenvalues have special properties:

```python
def symmetric_eigenproperties(A):
    """
    Demonstrate properties of symmetric matrix eigenvalues.
    
    Args:
        A: Symmetric matrix
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"All eigenvalues are real: {np.all(np.isreal(eigenvalues))}")
    print(f"Eigenvectors are orthogonal: {np.allclose(np.dot(eigenvectors.T, eigenvectors), np.eye(len(eigenvalues)))}")
    
    # Verify Av = λv for each eigenvalue-eigenvector pair
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = np.dot(A, eigenvec)
        lambda_v = eigenval * eigenvec
        print(f"Eigenvalue {i+1}: Av = λv is {np.allclose(Av, lambda_v)}")

# Example
A = np.array([[3, 1], [1, 3]])
symmetric_eigenproperties(A)
```

### Defective Matrices

Matrices that don't have a complete set of eigenvectors:

```python
def check_matrix_defectiveness(A):
    """
    Check if a matrix is defective (geometric multiplicity < algebraic multiplicity).
    
    Args:
        A: Square matrix
    
    Returns:
        is_defective, details
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    details = {}
    is_defective = False
    
    for i, eigenval in enumerate(eigenvalues):
        # Count algebraic multiplicity
        algebraic_mult = np.sum(np.abs(eigenvalues - eigenval) < 1e-10)
        
        # Count geometric multiplicity (rank of null space)
        null_space = A - eigenval * np.eye(A.shape[0])
        geometric_mult = A.shape[0] - np.linalg.matrix_rank(null_space)
        
        details[eigenval] = {
            'algebraic': algebraic_mult,
            'geometric': geometric_mult
        }
        
        if geometric_mult < algebraic_mult:
            is_defective = True
    
    return is_defective, details

# Example of defective matrix
A = np.array([[2, 1], [0, 2]])  # Jordan block
is_defective, details = check_matrix_defectiveness(A)
print(f"Matrix is defective: {is_defective}")
print(f"Details: {details}")
```

## Numerical Considerations

### Conditioning

Eigenvalue problems can be ill-conditioned:

```python
def eigenvalue_conditioning(A):
    """
    Analyze the conditioning of eigenvalue problem.
    
    Args:
        A: Square matrix
    
    Returns:
        condition_number, sensitivity_analysis
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Condition number of eigenvalue problem
    condition_numbers = []
    for i, eigenval in enumerate(eigenvalues):
        # Sensitivity is related to the angle between left and right eigenvectors
        left_eigenvec = np.linalg.eig(A.T)[1][:, i]
        right_eigenvec = eigenvectors[:, i]
        
        # Normalize
        left_eigenvec = left_eigenvec / np.linalg.norm(left_eigenvec)
        right_eigenvec = right_eigenvec / np.linalg.norm(right_eigenvec)
        
        # Condition number for this eigenvalue
        cond_num = 1 / abs(np.dot(left_eigenvec, right_eigenvec))
        condition_numbers.append(cond_num)
    
    return condition_numbers, eigenvalues

# Example
A = np.array([[1, 1000], [0, 1.001]])  # Nearly defective
condition_numbers, eigenvalues = eigenvalue_conditioning(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Condition numbers: {condition_numbers}")
```

## Summary

Eigenvalues and eigenvectors are fundamental concepts in linear algebra with wide applications:

1. **Definition**: Solutions to Av = λv
2. **Computation**: Characteristic polynomial, iterative methods
3. **Properties**: Trace, determinant, symmetry implications
4. **Applications**: PCA, spectral clustering, stability analysis
5. **Numerical issues**: Conditioning, defective matrices

Understanding these concepts is essential for advanced topics in machine learning, signal processing, and scientific computing.
