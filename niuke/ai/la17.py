from typing import List, Union
import numpy as np

def calculate_eigenvalues(matrix: List[List[Union[int, float]]]) -> List[float]:
    eigenvalues = np.linalg.eig(matrix)[0]
    return eigenvalues.tolist()

def main():
    matrix = eval(input())
    result = calculate_eigenvalues(matrix)
    print(result)

if __name__ == "__main__":
    main()
