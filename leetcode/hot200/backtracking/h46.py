"""
46. 全排列
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

题目链接：https://leetcode.cn/problems/permutations/

示例 1:
输入: nums = [1,2,3]
输出: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

示例 2:
输入: nums = [0,1]
输出: [[0,1],[1,0]]

示例 3:
输入: nums = [1]
输出: [[1]]

提示：
- 1 <= nums.length <= 6
- -10 <= nums[i] <= 10
- nums 中的所有整数 互不相同
"""
from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        def permute(nums):
            if len(nums) == 1:
                return [nums]
            result = []
            for i in range(len(nums)):
                for p in permute(nums[:i] + nums[i+1:]):
                    result.append([nums[i]] + p)
            return result
        return permute(nums)


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 2, 3]
    result = solution.permute(nums)
    expected = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [0, 1]
    result = solution.permute(nums)
    expected = [[0,1],[1,0]]
    assert len(result) == len(expected)
    
    # 测试用例3
    nums = [1]
    result = solution.permute(nums)
    expected = [[1]]
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
