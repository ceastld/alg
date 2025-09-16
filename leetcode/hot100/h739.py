"""
LeetCode 739. Daily Temperatures

题目描述：
给定一个整数数组temperatures，表示每天的温度，返回一个数组answer，其中answer[i]是指对于第i天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用0来代替。

示例：
temperatures = [73,74,75,71,69,72,76,73]
输出：[1,1,4,2,1,1,0,0]

数据范围：
- 1 <= temperatures.length <= 10^5
- 30 <= temperatures[i] <= 100
"""

class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        # Use monotonic stack to find next greater element
        # Stack stores indices of temperatures
        stack = []
        result = [0] * len(temperatures)
        
        for i, temp in enumerate(temperatures):
            # While current temperature is greater than stack top
            while stack and temperatures[stack[-1]] < temp:
                # Pop the index and calculate days difference
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            
            # Push current index to stack
            stack.append(i)
        
        return result
        