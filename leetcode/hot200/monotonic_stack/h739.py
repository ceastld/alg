"""
739. 每日温度
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

题目链接：https://leetcode.cn/problems/daily-temperatures/

示例 1:
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]

示例 2:
输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]

示例 3:
输入: temperatures = [30,60,90]
输出: [1,1,0]

提示：
- 1 <= temperatures.length <= 10^5
- 30 <= temperatures[i] <= 100
"""

from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        res = [0] * len(temperatures)
        for i, temp in enumerate(temperatures):
            while stack and temp > temperatures[stack[-1]]:
                res[stack[-1]] = i - stack[-1]
                stack.pop()
            stack.append(i)
        return res


def main():
    """测试用例"""
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
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
