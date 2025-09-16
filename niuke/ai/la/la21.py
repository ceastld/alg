import numpy as np

def solve_jacobi(A, b, n):
    """
    Solve linear system Ax = b using Jacobi method
    
    Args:
        A: coefficient matrix (list of lists)
        b: right-hand side vector (list)
        n: number of iterations (int)
    
    Returns:
        approximate solution x as list
    """
    # Convert to numpy arrays for easier computation
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Get dimensions
    size = len(b)
    
    # Initialize solution vector x with zeros
    x = np.zeros(size)
    
    # Perform n iterations
    for iteration in range(n):
        x_new = np.zeros(size)
        
        # Jacobi iteration formula: x_i^(k+1) = (b_i - sum(a_ij * x_j^(k))) / a_ii
        for i in range(size):
            # Calculate sum of off-diagonal terms
            sum_term = 0.0
            for j in range(size):
                if i != j:
                    sum_term += A[i, j] * x[j]
            
            # Update x_i
            x_new[i] = (b[i] - sum_term) / A[i, i]
        
        # Update x for next iteration
        x = x_new.copy()
    
    # Round to 4 decimal places and return as list
    return [round(val, 4) for val in x]


# 主程序
if __name__ == "__main__":
    # 输入
    ndarrayA = input()
    ndarrayB = input()
    n = input()

    # 处理输入
    import ast
    A = ast.literal_eval(ndarrayA)
    b = ast.literal_eval(ndarrayB)
    n = int(n)

    # 调用函数计算
    output = solve_jacobi(A,b,n)
    
    # 输出结果
    print(output)
