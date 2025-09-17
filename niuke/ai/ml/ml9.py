import numpy as np

def one_hot_encode(x, n_col=None):
    """
    One-hot encode categorical data
    
    Args:
        x: 1D numpy array containing integer values to encode
        n_col: Optional parameter specifying number of columns (categories)
               If not provided, automatically set to max value in input array + 1
    
    Returns:
        2D numpy array where:
        - Number of rows equals length of input array
        - Number of columns equals n_col or auto-determined number of categories
        - Each row has only one 1, rest are 0
    """
    # Convert to numpy array if not already
    x = np.array(x)
    
    # Determine number of columns if not provided
    if n_col is None:
        n_col = np.max(x) + 1
    
    # Create one-hot encoded matrix
    one_hot = np.zeros((len(x), n_col))
    
    # Set 1 at appropriate positions
    for i, val in enumerate(x):
        one_hot[i, val] = 1
    
    return one_hot

def to_categorical(x, n_col=None):
    """
    Alias for one_hot_encode for compatibility
    """
    return one_hot_encode(x, n_col)
    

if __name__ == "__main__":
    x = np.array(eval(input()))
    print(one_hot_encode(x))

