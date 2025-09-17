def model_fit_quality(training_accuracy, test_accuracy):
    """
    基于训练和测试准确率,确定模型是否过拟合、欠拟合或拟合良好，返回1、-1、0。
    
    判断规则：
    - 当训练准确率显著高于测试准确率（差异>=0.2）时，模型可能过拟合，返回1
    - 当训练和测试准确率均低于0.7时，模型可能欠拟合，返回-1
    - 当训练准确率与测试准确率差异不大时，模型表现良好，返回0
    
    :param training_accuracy: float, 训练准确率 (0 <= training_accuracy <= 1)
    :param test_accuracy: float, 测试准确率 (0 <= test_accuracy <= 1)
    :return: int, 1(过拟合)、-1(欠拟合)、0(表现良好)
    """
    # 计算训练准确率和测试准确率的差异
    accuracy_diff = training_accuracy - test_accuracy
    
    # 判断过拟合：训练准确率显著高于测试准确率（差异>=0.2）
    if accuracy_diff >= 0.2:
        return 1
    
    # 判断欠拟合：训练和测试准确率均低于0.7
    if training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    
    # 其他情况：模型表现良好
    return 0

if __name__ == "__main__":
    training_accuracy, test_accuracy = map(float, input().split())
    print(model_fit_quality(training_accuracy, test_accuracy))
