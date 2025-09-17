import numpy as np


def rmse(y_true, y_pred):
    d = y_true - y_pred
    result = np.sqrt(np.sum(d * d)/np.size(y_true))
    return round(result, 3)


if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    print(f"{rmse(y_true, y_pred):.3f}")
