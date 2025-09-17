import numpy as np


def calculate_loss(real_values, predicted_values, delta):
    # MSE: Mean Squared Error
    mse = np.mean((real_values - predicted_values) ** 2)
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(real_values - predicted_values))
    
    # Huber Loss: 当误差小于等于delta时使用MSE，否则使用MAE
    huber_loss = np.where(np.abs(real_values - predicted_values) <= delta, mse, mae)
    
    # Cosine Loss
    cosine_loss = 1 - np.dot(real_values, predicted_values) / (np.linalg.norm(real_values) * np.linalg.norm(predicted_values))
    
    return mse, mae, np.mean(huber_loss), cosine_loss


# 从标准输入读取数据
n = int(input())
real_values = []
predicted_values = []

for _ in range(n):
    real, predicted = map(float, input().split())
    real_values.append(real)
    predicted_values.append(predicted)

delta = float(input())  # 读取阈值

# 调用计算损失函数的函数
results = calculate_loss(np.array(real_values), np.array(predicted_values), delta)
# 输出结果
for value in results:
    print(f"{value:.6f}")
