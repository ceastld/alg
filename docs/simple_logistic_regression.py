import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class SimpleLogisticRegression:
    """
    Simple Logistic Regression implementation using numpy
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        """
        Initialize logistic regression model
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid output
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy cost function
        
        Args:
            X: Input features (m, n)
            y: Target labels (m,)
            
        Returns:
            Cost value
        """
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model
        
        Args:
            X: Training features (m, n)
            y: Training labels (m,)
        """
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            
            # Compute cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Backward propagation (compute gradients)
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions
        
        Args:
            X: Input features
            threshold: Decision threshold
            
        Returns:
            Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self) -> None:
        """
        Plot the cost function history
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()


def generate_sample_data(n_samples: int = 100, n_features: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample binary classification data
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
    """
    np.random.seed(42)
    
    # Generate two classes with different means
    class_0 = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], n_samples // 2)
    class_1 = np.random.multivariate_normal([2, 2], [[1, -0.5], [-0.5, 1]], n_samples // 2)
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_decision_boundary(model: SimpleLogisticRegression, X: np.ndarray, y: np.ndarray) -> None:
    """
    Plot the decision boundary for 2D data
    
    Args:
        model: Trained logistic regression model
        X: Feature matrix
        y: Target labels
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Probability')
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter, label='True Label')
    
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def main():
    """
    Main function to demonstrate the logistic regression implementation
    """
    print("=== Simple Logistic Regression Demo ===\n")
    
    # Generate sample data
    print("1. Generating sample data...")
    X, y = generate_sample_data(n_samples=200, n_features=2)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    # Create and train the model
    print("\n2. Training the model...")
    model = SimpleLogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"   Training accuracy: {accuracy:.4f}")
    
    # Show final parameters
    print(f"\n4. Final parameters:")
    print(f"   Weights: {model.weights}")
    print(f"   Bias: {model.bias:.4f}")
    
    # Plot results
    print("\n5. Plotting results...")
    model.plot_cost_history()
    plot_decision_boundary(model, X, y)
    
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    main()
