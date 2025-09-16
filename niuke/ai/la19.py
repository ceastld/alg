def inverse_2x2(matrix) :
    import numpy as np
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    det = a * d - b * c
    if det == 0:
        return None
    inv_det = 1 / det
    return [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]]

# 主程序
if __name__ == "__main__":
    # 输入矩阵
    matrix_input = input()

    # 处理输入
    import ast
    matrix = ast.literal_eval(matrix_input)

    # 调用函数计算逆矩阵
    output = inverse_2x2(matrix)
    
    # 输出结果
    print(output)
 