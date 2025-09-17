import numpy as np


def linear_regression_gradient_descent(X, y, alpha, iterations):
    m, n = X.shape
    weights = np.zeros(n)
    # 补全代码
    for _ in range(iterations):
        predictions = np.dot(X, weights)
        # cost = np.mean((predictions - y) ** 2)
        dw = (1 / m) * np.dot(X.T, (predictions - y.flatten()))
        weights -= alpha * dw
        
    return np.round(weights.flatten(), 4)


# 主程序
if __name__ == "__main__":
    # 输入矩阵和向量
    matrix_inputx = input()
    array_y = input()
    alpha = input()
    iterations = input()

    # 处理输入
    import ast

    matrix = np.array(ast.literal_eval(matrix_inputx))
    y = np.array(ast.literal_eval(array_y)).reshape(-1, 1)
    alpha = float(alpha)
    iterations = int(iterations)

    # 调用函数计算逆矩阵
    output = linear_regression_gradient_descent(matrix, y, alpha, iterations)

    # 输出结果
    print(output)
