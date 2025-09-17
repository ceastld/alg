import numpy as np


def gini_impurity(y: list[int]) -> float:
    d = {}
    for n in y:
        d[n] = d.get(n, 0) + 1
    p = []
    for k, v in d.items():
        p.append(v / len(y))
    p = np.array(p)
    res = 1 - np.dot(p, p)
    return round(res, 3)


if __name__ == "__main__":
    y = eval(input())
    print(f"{gini_impurity(y):.3f}")
