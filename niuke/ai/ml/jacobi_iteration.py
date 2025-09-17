import numpy as np
from typing import Tuple

def jacobi_iteration(A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Perform one Jacobi iteration for solving linear system Ax = b
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n,)
        x0: Initial guess (n,)
    
    Returns:
        x1: Updated solution after one Jacobi iteration
    """
    n = len(b)
    x1 = np.zeros_like(x0)
    
    for i in range(n):
        # Jacobi formula: x_i^(k+1) = (b_i - sum(A_ij * x_j^(k) for j != i)) / A_ii
        sum_term = 0
        for j in range(n):
            if j != i:
                sum_term += A[i, j] * x0[j]
        
        x1[i] = (b[i] - sum_term) / A[i, i]
    
    return x1

def solve_linear_system():
    """
    Solve the given linear system using Jacobi iteration
    System: { 4x + y = 6, x + 3y = 3 }
    Initial solution: (x0, y0) = (0, 0)
    """
    # Define the coefficient matrix A and right-hand side vector b
    A = np.array([[4, 1],
                  [1, 3]])
    
    b = np.array([6, 3])
    
    # Initial solution
    x0 = np.array([0.0, 0.0])
    
    print("Linear system:")
    print("4x + y = 6")
    print("x + 3y = 3")
    print(f"Initial solution: (x0, y0) = ({x0[0]}, {x0[1]})")
    print()
    
    # Perform one Jacobi iteration
    x1 = jacobi_iteration(A, b, x0)
    
    print("After one Jacobi iteration:")
    print(f"(x1, y1) = ({x1[0]:.1f}, {x1[1]:.1f})")
    print()
    
    # Verify the solution
    print("Verification:")
    print(f"4*{x1[0]:.1f} + {x1[1]:.1f} = {4*x1[0] + x1[1]:.1f} (should be 6)")
    print(f"{x1[0]:.1f} + 3*{x1[1]:.1f} = {x1[0] + 3*x1[1]:.1f} (should be 3)")
    print()
    
    # Check against the given options
    options = {
        'A': (0, 0),
        'B': (2, 0.5),
        'C': (1, 1),
        'D': (1.5, 1)
    }
    
    print("Comparing with given options:")
    for option, (x_opt, y_opt) in options.items():
        if abs(x1[0] - x_opt) < 0.1 and abs(x1[1] - y_opt) < 0.1:
            print(f"✓ Option {option}: ({x_opt}, {y_opt}) - MATCHES!")
        else:
            print(f"✗ Option {option}: ({x_opt}, {y_opt})")
    
    return x1

if __name__ == "__main__":
    result = solve_linear_system()
