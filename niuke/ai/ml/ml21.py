from collections import Counter

def confusion_matrix(data):
    """
    Generate confusion matrix for binary classification
    
    Confusion matrix is a 2x2 matrix that shows the performance of a classification model:
    [[TP, FN],
     [FP, TN]]
    
    where:
    - TP (True Positive): Correctly predicted positive samples
    - FN (False Negative): Incorrectly predicted negative samples  
    - FP (False Positive): Incorrectly predicted positive samples
    - TN (True Negative): Correctly predicted negative samples
    
    Args:
        data: List of lists, where each inner list contains [y_true, y_pred]
    
    Returns:
        List of lists: 2x2 confusion matrix [[TP, FN], [FP, TN]]
    """
    # Initialize counters for each category
    TP = 0  # True Positive: y_true=1, y_pred=1
    FN = 0  # False Negative: y_true=1, y_pred=0
    FP = 0  # False Positive: y_true=0, y_pred=1
    TN = 0  # True Negative: y_true=0, y_pred=0
    
    # Count each type of prediction
    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 1 and y_pred == 0:
            FN += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        elif y_true == 0 and y_pred == 0:
            TN += 1
    
    return [[TP, FN], [FP, TN]]

if __name__ == "__main__":
    data = eval(input())
    print(confusion_matrix(data))
