import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for binary classification
    
    F-Score is the weighted harmonic mean of precision and recall.
    Formula: F_β = (1 + β²) * (precision * recall) / (β² * precision + recall)
    
    Args:
        y_true: 1D numpy array, true binary labels (0 or 1)
        y_pred: 1D numpy array, predicted binary labels (0 or 1)
        beta: float, weight parameter for precision vs recall
    
    Returns:
        float: F-Score value rounded to 3 decimal places
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    beta = float(beta)
    
    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F-Score
    if precision == 0 and recall == 0:
        score = 0.0
    else:
        score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    return round(score, 3)

if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    beta = float(input())
    print(f"{f_score(y_true, y_pred, beta):.3f}")
