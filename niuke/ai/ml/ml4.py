import numpy as np

def preprocess_data():
    n = int(input())
    data = []
    for _ in range(n):
        data.append(float(input().strip()))
    
    # Convert to numpy array
    data = np.array(data)
    
    # Step 1: Calculate mean using non-missing values (excluding -1)
    non_missing_mask = data != -1
    mean_value = np.mean(data[non_missing_mask])
    
    # Step 2: Fill missing values with mean
    data_filled = data.copy()
    data_filled[data == -1] = mean_value
    
    # Step 3: Remove outliers (values > 800 or < 200)
    valid_mask = (data_filled >= 200) & (data_filled <= 800)
    processed_data = data_filled[valid_mask]
    
    return processed_data

def main():
    ans = preprocess_data()
    for i in range(len(ans)):
        print(f"{ans[i]:.4f}")

if __name__ == '__main__':
    main()