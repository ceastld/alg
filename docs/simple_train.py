import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X: np.ndarray, y: np.ndarray, 
                            learning_rate: float = 0.01, 
                            max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Train logistic regression model using gradient descent
    
    Args:
        X: Training features (m, n)
        y: Training labels (m,)
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        
    Returns:
        weights: Learned weights (n,)
        bias: Learned bias (scalar)
    """
    m, n = X.shape
    
    # Initialize parameters
    weights = np.zeros(n)
    bias = 0.0
    
    # Gradient descent
    for i in range(max_iterations):
        # Forward propagation
        z = np.dot(X, weights) + bias
        h = sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            # Compute cost for monitoring
            epsilon = 1e-15
            h_clipped = np.clip(h, epsilon, 1 - epsilon)
            cost = -(1/m) * np.sum(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
            print(f"Iteration {i}, Cost: {cost:.4f}")
    
    return weights, bias


def predict(X: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5) -> np.ndarray:
    """
    Make predictions using trained model
    
    Args:
        X: Input features
        weights: Learned weights
        bias: Learned bias
        threshold: Decision threshold
        
    Returns:
        Binary predictions
    """
    z = np.dot(X, weights) + bias
    probabilities = sigmoid(z)
    return (probabilities >= threshold).astype(int)


# Example usage
if __name__ == "__main__":
    # Generate simple sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print("Training logistic regression...")
    weights, bias = train_logistic_regression(X, y, learning_rate=0.1, max_iterations=500)
    
    print(f"\nFinal weights: {weights}")
    print(f"Final bias: {bias:.4f}")
    
    # Test predictions
    y_pred = predict(X, weights, bias)
    accuracy = np.mean(y_pred == y)
    print(f"Training accuracy: {accuracy:.4f}")
