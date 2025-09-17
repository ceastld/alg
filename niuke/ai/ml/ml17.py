import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (R²) coefficient of determination.
    
    R-squared measures the proportion of variance in the dependent variable
    that is predictable from the independent variable(s).
    
    Formula: R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = Σ(y_true - y_pred)² (sum of squares of residuals)
    - SS_tot = Σ(y_true - y_mean)² (total sum of squares)
    
    Args:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values
    
    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the mean of true values
    y_mean = np.mean(y_true)
    
    # Calculate sum of squares of residuals (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate total sum of squares (SS_tot)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Calculate R-squared
    if ss_tot == 0:
        # If all true values are the same, R² is undefined
        # Return 0 in this case
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
    
    return round(r2, 3)


if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    print(f"{r_squared(y_true, y_pred):.3f}")
