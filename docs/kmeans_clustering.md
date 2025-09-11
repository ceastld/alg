# K-means聚类完整数学推导

## 1. K-means基本概念

### 1.1 核心思想
K-means是一种无监督学习算法，用于将数据分成K个簇。算法的目标是最小化簇内平方和（Within-Cluster Sum of Squares, WCSS）。

### 1.2 数学表示
给定数据集 $X = \{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$，其中 $x^{(i)} \in \mathbb{R}^d$，K-means算法寻找K个聚类中心 $\mu_1, \mu_2, \ldots, \mu_K$，使得：

$$\min_{C, \mu} \sum_{i=1}^m \sum_{k=1}^K w_{ik} ||x^{(i)} - \mu_k||^2$$

其中：
- $w_{ik}$ 是指示变量，如果 $x^{(i)}$ 属于簇 $k$，则 $w_{ik} = 1$，否则 $w_{ik} = 0$
- $C = \{C_1, C_2, \ldots, C_K\}$ 是簇的集合
- $\mu = \{\mu_1, \mu_2, \ldots, \mu_K\}$ 是聚类中心

### 1.3 约束条件
每个数据点必须属于且仅属于一个簇：

$$\sum_{k=1}^K w_{ik} = 1, \quad \forall i$$

## 2. 目标函数推导

### 2.1 簇内平方和（WCSS）
对于簇 $k$，簇内平方和为：

$$WCSS_k = \sum_{x^{(i)} \in C_k} ||x^{(i)} - \mu_k||^2$$

总的簇内平方和为：

$$WCSS = \sum_{k=1}^K WCSS_k = \sum_{k=1}^K \sum_{x^{(i)} \in C_k} ||x^{(i)} - \mu_k||^2$$

### 2.2 目标函数
K-means的目标函数为：

$$J(C, \mu) = \sum_{k=1}^K \sum_{x^{(i)} \in C_k} ||x^{(i)} - \mu_k||^2$$

$$= \sum_{i=1}^m \sum_{k=1}^K w_{ik} ||x^{(i)} - \mu_k||^2$$

## 3. 优化算法

### 3.1 交替优化
K-means使用交替优化策略，分别优化簇分配和聚类中心：

#### 步骤1：固定聚类中心，优化簇分配
对于每个数据点 $x^{(i)}$，将其分配到最近的聚类中心：

$$c^{(i)} = \arg\min_k ||x^{(i)} - \mu_k||^2$$

其中 $c^{(i)}$ 是 $x^{(i)}$ 所属的簇索引。

#### 步骤2：固定簇分配，优化聚类中心
对于每个簇 $k$，计算其聚类中心：

$$\mu_k = \frac{1}{|C_k|} \sum_{x^{(i)} \in C_k} x^{(i)}$$

其中 $|C_k|$ 是簇 $k$ 中数据点的数量。

### 3.2 算法收敛性
K-means算法保证收敛，因为：
1. 目标函数在每次迭代后都会减少或保持不变
2. 目标函数有下界（非负）
3. 可能的簇分配数量是有限的

## 4. 初始化策略

### 4.1 随机初始化
随机选择K个数据点作为初始聚类中心。

### 4.2 K-means++初始化
K-means++是一种改进的初始化方法：

1. 随机选择第一个聚类中心
2. 对于每个后续的聚类中心，选择距离已选中心最远的点，概率与距离平方成正比

选择概率：

$$P(x^{(i)}) = \frac{D(x^{(i)})^2}{\sum_{j=1}^m D(x^{(j)})^2}$$

其中 $D(x^{(i)})$ 是 $x^{(i)}$ 到最近已选聚类中心的距离。

## 5. 距离度量

### 5.1 欧几里得距离
最常用的距离度量：

$$d(x^{(i)}, \mu_k) = ||x^{(i)} - \mu_k||_2 = \sqrt{\sum_{j=1}^d (x_j^{(i)} - \mu_{k,j})^2}$$

### 5.2 曼哈顿距离
$$d(x^{(i)}, \mu_k) = ||x^{(i)} - \mu_k||_1 = \sum_{j=1}^d |x_j^{(i)} - \mu_{k,j}|$$

### 5.3 余弦距离
$$d(x^{(i)}, \mu_k) = 1 - \frac{x^{(i)} \cdot \mu_k}{||x^{(i)}|| ||\mu_k||}$$

## 6. 算法复杂度

### 6.1 时间复杂度
- 每次迭代：$O(mKd)$
- 总复杂度：$O(tmKd)$，其中 $t$ 是迭代次数

### 6.2 空间复杂度
$O(m + Kd)$

## 7. Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.datasets import make_blobs
import random

class KMeans:
    """
    K-means clustering implementation
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, 
                 init: str = 'random', random_state: Optional[int] = None):
        """
        Initialize K-means clustering
        
        Args:
            n_clusters: Number of clusters
            max_iters: Maximum number of iterations
            init: Initialization method ('random' or 'kmeans++')
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids"""
        if self.init == 'random':
            return self._random_init(X)
        elif self.init == 'kmeans++':
            return self._kmeans_plus_plus_init(X)
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
    
    def _random_init(self, X: np.ndarray) -> np.ndarray:
        """Random initialization"""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            centroids[k] = X[np.random.randint(n_samples)]
        
        return centroids
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization"""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(k):
                    dist = np.sum((X[i] - centroids[j]) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.random()
            
            for i in range(n_samples):
                if r <= cumulative_probabilities[i]:
                    centroids[k] = X[i]
                    break
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid"""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = np.sum((X[i] - self.centroids) ** 2, axis=1)
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids based on current cluster assignments"""
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep the old centroid
                centroids[k] = self.centroids[k]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute within-cluster sum of squares (inertia)"""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-means clustering to the data
        
        Args:
            X: Input data (m, n)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iterative optimization
        for iteration in range(self.max_iters):
            # Assign points to clusters
            old_labels = self.labels.copy() if self.labels is not None else None
            self.labels = self._assign_clusters(X)
            
            # Check for convergence
            if old_labels is not None and np.array_equal(self.labels, old_labels):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels)
            
            # Compute inertia
            self.inertia_ = self._compute_inertia(X, self.labels)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Inertia: {self.inertia_:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels

def generate_sample_data(n_samples: int = 300, n_features: int = 2, 
                        n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample clustering data"""
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, 
                          n_features=n_features, random_state=42)
    return X, y_true

def plot_clustering_results(X: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray, centroids: np.ndarray,
                          title: str = "K-means Clustering") -> None:
    """Plot clustering results"""
    plt.figure(figsize=(15, 5))
    
    # True clusters
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Predicted clusters
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', 
               s=200, linewidths=3, label='Centroids')
    plt.title('Predicted Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter)
    
    # Comparison
    plt.subplot(1, 3, 3)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(len(np.unique(y_true))):
        mask_true = y_true == i
        mask_pred = y_pred == i
        plt.scatter(X[mask_true, 0], X[mask_true, 1], 
                   c=colors[i % len(colors)], marker='o', alpha=0.5, 
                   label=f'True Cluster {i}')
        plt.scatter(X[mask_pred, 0], X[mask_pred, 1], 
                   c=colors[i % len(colors)], marker='s', alpha=0.5, 
                   label=f'Pred Cluster {i}')
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', 
               s=200, linewidths=3, label='Centroids')
    plt.title('Comparison')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate clustering performance"""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    return {
        'Adjusted Rand Index': ari,
        'Normalized Mutual Information': nmi
    }

def main():
    """Main function to demonstrate K-means clustering"""
    print("=== K-means Clustering Demo ===\n")
    
    # Generate sample data
    print("1. Generating sample data...")
    X, y_true = generate_sample_data(n_samples=300, n_features=2, n_clusters=3)
    print(f"   Data shape: X={X.shape}")
    print(f"   True clusters: {len(np.unique(y_true))}")
    
    # Random initialization
    print("\n2. K-means with random initialization...")
    kmeans_random = KMeans(n_clusters=3, max_iters=100, init='random', random_state=42)
    y_pred_random = kmeans_random.fit_predict(X)
    
    print(f"   Final inertia: {kmeans_random.inertia_:.4f}")
    print(f"   Centroids:\n{kmeans_random.centroids}")
    
    # Evaluate clustering
    metrics_random = evaluate_clustering(y_true, y_pred_random)
    print(f"   Adjusted Rand Index: {metrics_random['Adjusted Rand Index']:.4f}")
    print(f"   Normalized Mutual Information: {metrics_random['Normalized Mutual Information']:.4f}")
    
    # K-means++ initialization
    print("\n3. K-means with K-means++ initialization...")
    kmeans_plus = KMeans(n_clusters=3, max_iters=100, init='kmeans++', random_state=42)
    y_pred_plus = kmeans_plus.fit_predict(X)
    
    print(f"   Final inertia: {kmeans_plus.inertia_:.4f}")
    print(f"   Centroids:\n{kmeans_plus.centroids}")
    
    # Evaluate clustering
    metrics_plus = evaluate_clustering(y_true, y_pred_plus)
    print(f"   Adjusted Rand Index: {metrics_plus['Adjusted Rand Index']:.4f}")
    print(f"   Normalized Mutual Information: {metrics_plus['Normalized Mutual Information']:.4f}")
    
    # Plot results
    print("\n4. Plotting results...")
    plot_clustering_results(X, y_true, y_pred_random, kmeans_random.centroids, 
                          "K-means with Random Initialization")
    plot_clustering_results(X, y_true, y_pred_plus, kmeans_plus.centroids, 
                          "K-means with K-means++ Initialization")
    
    # Elbow method for choosing K
    print("\n5. Elbow method for choosing optimal K...")
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, max_iters=100, init='kmeans++', random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Demo completed ===")

if __name__ == "__main__":
    main()
```

## 8. 总结

K-means聚类通过以下步骤实现数据分组：

1. **初始化**：选择K个初始聚类中心（随机或K-means++）
2. **分配**：将每个数据点分配到最近的聚类中心
3. **更新**：重新计算每个簇的聚类中心
4. **迭代**：重复步骤2和3直到收敛
5. **输出**：返回最终的聚类中心和簇分配

K-means具有简单、高效、可扩展的特点，是聚类任务中最常用的算法之一。但需要注意选择合适的K值和初始化策略，以及处理不同形状和密度的簇。
