from typing import List, Union
import math

def calculate_eigenvalues_2d(matrix: List[List[Union[int, float]]]) -> List[float]:
    """
    Calculate eigenvalues for 2x2 matrix using direct formula.
    For matrix [[a, b], [c, d]], eigenvalues are:
    λ = (a + d ± sqrt((a + d)² - 4(ad - bc))) / 2
    """
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]
    
    # Calculate trace and determinant
    trace = a + d
    det = a * d - b * c
    
    # Calculate discriminant
    discriminant = trace * trace - 4 * det
    
    if discriminant >= 0:
        # Real eigenvalues
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        return [lambda1, lambda2]
    else:
        # Complex eigenvalues
        sqrt_disc = math.sqrt(-discriminant)
        real_part = trace / 2
        imag_part = sqrt_disc / 2
        return [complex(real_part, imag_part), complex(real_part, -imag_part)]

def main():
    matrix = eval(input())
    result = calculate_eigenvalues_2d(matrix)
    print(result)

if __name__ == "__main__":
    main()
