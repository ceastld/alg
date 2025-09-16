import numpy as np

def matrixmul(a,b):
    # 补全代码
    if len(a[0]) != len(b):
        return -1
    return np.dot(a,b).tolist()

# 主程序
if __name__ == "__main__":
    # 输入矩阵
    matrix_inputa = input()
    matrix_inputb = input()

    # 处理输入
    import ast
    matrixa = ast.literal_eval(matrix_inputa)
    matrixb = ast.literal_eval(matrix_inputb)

    # 调用函数计算逆矩阵
    output = matrixmul(matrixa,matrixb)
    
    # 输出结果
    print(output)
