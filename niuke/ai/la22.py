import numpy as np

def calculate_correlation_matrix(X, Y=None):
    """
    Calculate correlation matrix between variables.
    
    Args:
        X: First data matrix, shape (n_samples, n_features_1)
        Y: Second data matrix (optional), shape (n_samples, n_features_2)
            If None, calculate correlation matrix of X with itself
    
    Returns:
        Correlation matrix:
        - If only X: shape (n_features_1, n_features_1)
        - If Y provided: shape (n_features_1, n_features_2)
    """
    if Y is None:
        # Calculate correlation matrix of X with itself
        return np.corrcoef(X.T)
    else:
        # Calculate correlation matrix between X and Y
        # Concatenate X and Y, then calculate correlation matrix
        combined = np.hstack([X, Y])
        corr_matrix = np.corrcoef(combined.T)
        
        # Extract the cross-correlation part
        n_features_x = X.shape[1]
        n_features_y = Y.shape[1]
        
        return corr_matrix[:n_features_x, n_features_x:]

if __name__ == "__main__":
    X = np.array(eval(input()))
    print(calculate_correlation_matrix(X))
