def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert dense matrix to Compressed Row Sparse (CSR) format.
    
    Args:
        dense_matrix: 2D list representing the dense matrix
        
    Returns:
        tuple: (values, column_indices, row_pointer)
    """
    values = []
    column_indices = []
    row_pointer = [0]  # First row starts at index 0
    
    for row in dense_matrix:
        row_start = len(values)  # Current position in values array
        
        for col_idx, value in enumerate(row):
            if value != 0:  # Only store non-zero elements
                values.append(value)
                column_indices.append(col_idx)
        
        row_pointer.append(len(values))  # End position of current row
    
    return values, column_indices, row_pointer


if __name__ == "__main__":
    dense_matrix = eval(input())
    vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
    print(vals)
    print(col_idx)
    print(row_ptr)