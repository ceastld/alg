import numpy as np


def transform_matrix(A: list, T: list, S: list) -> list:
    """
    Transform matrix A using T^(-1) * A * S

    Args:
        A: Input matrix A
        T: Invertible matrix T
        S: Invertible matrix S

    Returns:
        Transformed matrix as nested list, or -1 if matrices are not invertible
    """
    try:
        # Convert to numpy arrays
        A_np = np.array(A, dtype=float)
        T_np = np.array(T, dtype=float)
        S_np = np.array(S, dtype=float)

        # Check if T and S are invertible by computing determinants
        det_T = np.linalg.det(T_np)
        det_S = np.linalg.det(S_np)

        # If determinant is close to zero, matrix is not invertible
        if abs(det_T) < 1e-10 or abs(det_S) < 1e-10:
            return -1

        # Compute T^(-1) * A * S
        T_inv = np.linalg.inv(T_np)
        result = T_inv @ A_np @ S_np

        # Convert back to nested list and round to 3 decimal places
        result_list = result.tolist()
        result_rounded = [[round(x, 3) for x in row] for row in result_list]

        return result_rounded

    except np.linalg.LinAlgError:
        # Handle case where matrix is singular (not invertible)
        return -1
    except Exception:
        # Handle any other errors
        return -1


# 主程序
if __name__ == "__main__":
    # 输入
    ndarrayA = input()
    ndarrayT = input()
    ndarrayS = input()

    # 处理输入
    import ast

    A = ast.literal_eval(ndarrayA)
    T = ast.literal_eval(ndarrayT)
    S = ast.literal_eval(ndarrayS)

    # 调用函数计算
    output = transform_matrix(A, T, S)

    # 输出结果
    print(output)
