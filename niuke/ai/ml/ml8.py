import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size] if y is not None else None
        yield [X_batch, y_batch]
    
if __name__ == "__main__":
    X = np.array(eval(input()))
    y = np.array(eval(input()))
    batch_size = int(input())
    print(list(batch_iterator(X, y, batch_size)))

