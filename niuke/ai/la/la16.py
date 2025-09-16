import numpy as np


def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
            data (list[float]): A list of numerical values to transform.
            degree (int): The degree of the polynomial expansion.

    Returns:
            list[list[float]]: A nested list where each inner list represents the transformed features of a data point.
    """
    if len(data) == 0 or degree < 0:
        return []
    return np.power(np.array(data, dtype=float)[:, np.newaxis], np.arange(0, degree + 1)).tolist()


if __name__ == "__main__":
    data = eval(input())
    degree = int(input())
    print(phi_transform(data, degree))
