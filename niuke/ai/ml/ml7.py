import numpy as np

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    return X[indices], y[indices]
    
if __name__ == "__main__":
    X = np.array(eval(input()))
    y = np.array(eval(input()))
    print(shuffle_data(X, y, 42))


