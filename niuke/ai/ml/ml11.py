import numpy as np

def ridge_loss(X, w, y_true, alpha):
    """
    Calculate Ridge regression loss (L2 regularized mean squared error)
    
    Args:
        X: 2D numpy array, feature matrix (n_samples, n_features)
        w: 1D numpy array, weight vector (n_features,)
        y_true: 1D numpy array, true target values (n_samples,)
        alpha: float, regularization parameter (lambda)
    
    Returns:
        float: Ridge loss value
    """
    # Convert to numpy arrays if not already
    X = np.array(X)
    w = np.array(w)
    y_true = np.array(y_true)
    alpha = float(alpha)
    
    # Calculate predictions: X @ w
    y_pred = X @ w
    
    # Calculate mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Calculate L2 regularization term (sum of squared weights)
    l2_penalty = alpha * np.sum(w ** 2)
    
    # Total ridge loss = MSE + L2 penalty
    loss = mse + l2_penalty
    
    return loss


if __name__ == "__main__":
    X = np.array(eval(input()))
    w = np.array(eval(input()))
    y_true = np.array(eval(input()))
    alpha = eval(input())
    print(ridge_loss(X, w, y_true, alpha))
