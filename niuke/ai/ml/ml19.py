import numpy as np

def jaccard_index(y_true, y_pred):
    """
    Calculate Jaccard Index for binary classification
    
    Jaccard Index measures the similarity between two sets by calculating
    the ratio of the intersection to the union of the sets.
    
    Formula: Jaccard Index = |A ∩ B| / |A ∪ B|
    where A and B are the sets of positive predictions and true labels
    
    Args:
        y_true: 1D numpy array, true binary labels (0 or 1)
        y_pred: 1D numpy array, predicted binary labels (0 or 1)
    
    Returns:
        float: Jaccard Index value rounded to 3 decimal places
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate intersection: elements that are 1 in both arrays
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate union: elements that are 1 in either array
    union = np.sum((y_true == 1) | (y_pred == 1))
    
    # Calculate Jaccard Index
    if union == 0:
        # If there are no positive elements in either array, Jaccard is undefined
        # Return 1.0 in this case (perfect similarity when both are all zeros)
        result = 1.0
    else:
        result = intersection / union
    
    return round(result, 3)

if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    print(f"{jaccard_index(y_true, y_pred):.3f}")
