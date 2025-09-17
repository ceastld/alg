import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score for classification predictions
    
    Args:
        y_true: 1D numpy array containing true class labels
        y_pred: 1D numpy array containing predicted class labels
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Calculate accuracy: number of correct predictions / total predictions
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    
    return accuracy
    
    
if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    print(accuracy_score(y_true, y_pred))
