"""
15. 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

题目链接：https://leetcode.cn/problems/3sum/

示例 1:
输入: nums = [-1,0,1,2,-1,-4]
输出: [[-1,-1,2],[-1,0,1]]

示例 2:
输入: nums = []
输出: []

示例 3:
输入: nums = [0]
输出: []

提示：
- 0 <= nums.length <= 3000
- -10^5 <= nums[i] <= 10^5
"""
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [-1, 0, 1, 2, -1, -4]
    result = solution.threeSum(nums)
    expected = [[-1, -1, 2], [-1, 0, 1]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = []
    result = solution.threeSum(nums)
    assert result == []
    
    # 测试用例3
    nums = [0]
    result = solution.threeSum(nums)
    assert result == []
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
