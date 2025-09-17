import numpy as np


def feature_scaling(data):
    # 补全代码
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    r1 = (data - mean) / std
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    r2 = (data - data_min) / (data_max - data_min)

    return np.round(r1, 4).tolist(), np.round(r2, 4).tolist()


# 主程序
if __name__ == "__main__":
    # 输入数组
    data = input()

    # 处理输入
    import ast

    data = ast.literal_eval(data)

    # 调用函数计算
    output = feature_scaling(data)

    # 输出结果
    print(output)
