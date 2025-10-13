"""
739. 每日温度 - 标准答案
"""
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        标准解法：单调栈
        
        解题思路：
        1. 使用单调栈维护温度递减的索引
        2. 遍历温度数组，对于每个温度：
           - 如果当前温度大于栈顶温度，则弹出栈顶元素并计算天数差
           - 将当前索引压入栈中
        3. 最后栈中剩余的元素都是没有更高温度的，设为0
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []  # 存储索引
        
        for i in range(n):
            # 当前温度大于栈顶温度时，弹出并计算天数差
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            
            # 将当前索引压入栈中
            stack.append(i)
        
        return result
    
    def dailyTemperatures_brute_force(self, temperatures: List[int]) -> List[int]:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 对于每个位置，向后查找第一个更高的温度
        2. 如果找到，记录天数差；否则设为0
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        n = len(temperatures)
        result = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if temperatures[j] > temperatures[i]:
                    result[i] = j - i
                    break
        
        return result
    
    def dailyTemperatures_optimized(self, temperatures: List[int]) -> List[int]:
        """
        优化解法：单调栈（从右到左）
        
        解题思路：
        1. 从右到左遍历温度数组
        2. 维护一个单调栈，存储温度递减的索引
        3. 对于每个位置，找到栈中第一个大于当前温度的位置
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []
        
        for i in range(n - 1, -1, -1):
            # 弹出所有小于等于当前温度的元素
            while stack and temperatures[stack[-1]] <= temperatures[i]:
                stack.pop()
            
            # 如果栈不为空，说明有更高的温度
            if stack:
                result[i] = stack[-1] - i
            
            # 将当前索引压入栈中
            stack.append(i)
        
        return result
    
    def dailyTemperatures_detailed(self, temperatures: List[int]) -> List[int]:
        """
        详细解法：单调栈（带详细注释）
        
        解题思路：
        1. 使用单调栈维护温度递减的索引序列
        2. 当遇到更高温度时，弹出栈中所有较低温度的索引
        3. 计算天数差并更新结果数组
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []  # 存储温度递减的索引
        
        for i in range(n):
            current_temp = temperatures[i]
            
            # 当前温度大于栈顶温度时，需要处理栈中所有较低温度
            while stack and current_temp > temperatures[stack[-1]]:
                # 弹出栈顶索引
                prev_index = stack.pop()
                # 计算天数差
                result[prev_index] = i - prev_index
            
            # 将当前索引压入栈中
            stack.append(i)
        
        return result
    
    def dailyTemperatures_alternative(self, temperatures: List[int]) -> List[int]:
        """
        替代解法：单调栈（使用deque）
        
        解题思路：
        1. 使用collections.deque作为栈
        2. 其他逻辑与标准解法相同
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        from collections import deque
        
        n = len(temperatures)
        result = [0] * n
        stack = deque()
        
        for i in range(n):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            
            stack.append(i)
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.dailyTemperatures([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0]
    
    # 测试用例2
    assert solution.dailyTemperatures([30,40,50,60]) == [1,1,1,0]
    
    # 测试用例3
    assert solution.dailyTemperatures([30,60,90]) == [1,1,0]
    
    # 测试用例4
    assert solution.dailyTemperatures([55,38,53,81,61,93,97,32,43,78]) == [3,1,1,2,1,1,0,1,1,0]
    
    # 测试用例5
    assert solution.dailyTemperatures([89,62,70,58,47,47,46,76,100,70]) == [8,1,5,4,3,2,1,1,0,0]
    
    # 测试用例6：边界情况
    assert solution.dailyTemperatures([1]) == [0]
    assert solution.dailyTemperatures([1,2]) == [1,0]
    assert solution.dailyTemperatures([2,1]) == [0,0]
    
    # 测试用例7：单调递增
    assert solution.dailyTemperatures([1,2,3,4,5]) == [1,1,1,1,0]
    
    # 测试用例8：单调递减
    assert solution.dailyTemperatures([5,4,3,2,1]) == [0,0,0,0,0]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
