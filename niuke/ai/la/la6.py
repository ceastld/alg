import numpy as np

def calculate_covariance_matrix(vectors):
    # Convert to numpy array and transpose to get features as columns
    data = np.array(vectors).T
    
    # Calculate mean of each feature
    mean = np.mean(data, axis=0)
    
    # Center the data (subtract mean)
    centered_data = data - mean
    
    # Calculate covariance matrix using population formula (divide by n)
    n = data.shape[0]
    cov_matrix = np.dot(centered_data.T, centered_data) / (n-1)
    
    return cov_matrix.tolist()

# 主程序
if __name__ == "__main__":
    # 输入
    ndarrayA = input()

    # 处理输入
    import ast
    A = ast.literal_eval(ndarrayA)

    # 调用函数计算
    output = calculate_covariance_matrix(A)
    
    # 输出结果
    print(output)

