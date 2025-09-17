import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """
    Perform linear regression using normal equation.
    
    Args:
        X: Feature matrix (list of lists) - already includes bias column
        y: Target vector (list)
    
    Returns:
        List containing [weight, bias] coefficients
    """
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Solve normal equation: X^T * X * theta = X^T * y
    # Using np.linalg.solve for better numerical stability
    XtX = X.T @ X
    Xty = X.T @ y
    
    try:
        theta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Matrix is singular, use least squares solution
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Extract weight and bias, round to 4 decimal places
    # theta[0] is weight, theta[1] is bias
    weight = round(float(theta[0]), 4)
    bias = round(float(theta[1]), 4) if len(theta) > 1 else 0.0
    
    # Convert -0.0 to 0.0 using abs() for cleaner code
    weight = abs(weight) if weight == 0.0 else weight
    bias = abs(bias) if bias == 0.0 else bias
    
    return [weight, bias]

if __name__ == "__main__":
    import ast
    x = np.array(ast.literal_eval(input()))
    y = np.array(ast.literal_eval(input())).reshape(-1, 1)

    # Perform linear regression
    coefficients = linear_regression_normal_equation(x, y)

    # Print the coefficients
    print(coefficients)
