import numpy as np

def dice_score(y_true, y_pred):
    """
    Calculate Dice coefficient for binary classification
    
    Dice coefficient is a measure of overlap between two sets, commonly used
    in image segmentation and classification tasks.
    
    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
    where A and B are the sets of positive predictions and true labels
    
    Args:
        y_true: 1D numpy array, true binary labels (0 or 1)
        y_pred: 1D numpy array, predicted binary labels (0 or 1)
    
    Returns:
        float: Dice coefficient value rounded to 3 decimal places
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate intersection: elements that are 1 in both arrays
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate the sum of positive elements in both arrays
    sum_positive = np.sum(y_true == 1) + np.sum(y_pred == 1)
    
    # Calculate Dice coefficient
    if sum_positive == 0:
        # If there are no positive elements in either array, Dice is undefined
        # Return 1.0 in this case (perfect similarity when both are all zeros)
        res = 1.0
    else:
        res = 2 * intersection / sum_positive
    
    return round(res, 3)

if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    print(f"{dice_score(y_true, y_pred):.3f}")
