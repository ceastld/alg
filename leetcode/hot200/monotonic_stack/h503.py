"""
503. 下一个更大元素II
给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。

数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。

题目链接：https://leetcode.cn/problems/next-greater-element-ii/

示例 1:
输入: nums = [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；
数字 2 找不到下一个更大的数； 
第二个 1 的下一个更大的数需要循环搜索，结果也是 2。

示例 2:
输入: nums = [1,2,3,4,3]
输出: [2,3,4,-1,4]

提示：
- 1 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
"""

from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        """
        标准解法：单调栈（循环数组）
        
        解题思路：
        1. 将数组扩展为两倍长度，模拟循环数组
        2. 使用单调栈维护递减序列
        3. 对于每个元素，找到其下一个更大元素
        4. 只处理前n个元素的结果
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # 遍历两倍长度的数组，模拟循环
        for i in range(2 * n):
            # 使用模运算获取实际索引
            actual_index = i % n
            current_num = nums[actual_index]
            
            # 当前元素大于栈顶元素时，弹出并更新结果
            while stack and current_num > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = current_num
            
            # 只处理前n个元素
            if i < n:
                stack.append(actual_index)
        
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.nextGreaterElements([1,2,1]) == [2,-1,2]
    
    # 测试用例2
    assert solution.nextGreaterElements([1,2,3,4,3]) == [2,3,4,-1,4]
    
    # 测试用例3
    assert solution.nextGreaterElements([1,1,1,1,1]) == [-1,-1,-1,-1,-1]
    
    # 测试用例4
    assert solution.nextGreaterElements([5,4,3,2,1]) == [-1,5,5,5,5]
    
    # 测试用例5
    assert solution.nextGreaterElements([1,2,3,4,5]) == [2,3,4,5,-1]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
