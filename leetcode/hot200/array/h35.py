"""
35. 搜索插入位置
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

题目链接：https://leetcode.cn/problems/search-insert-position/

示例 1:
输入: nums = [1,3,5,6], target = 5
输出: 2

示例 2:
输入: nums = [1,3,5,6], target = 2
输出: 1

示例 3:
输入: nums = [1,3,5,6], target = 7
输出: 4

提示：
- 1 <= nums.length <= 10^4
- -10^4 <= nums[i] <= 10^4
- nums 为无重复元素的升序排列数组
- -10^4 <= target <= 10^4
"""
from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 3, 5, 6]
    target = 5
    assert solution.searchInsert(nums, target) == 2
    
    # 测试用例2
    nums = [1, 3, 5, 6]
    target = 2
    assert solution.searchInsert(nums, target) == 1
    
    # 测试用例3
    nums = [1, 3, 5, 6]
    target = 7
    assert solution.searchInsert(nums, target) == 4
    
    # 测试用例4
    nums = [1, 3, 5, 6]
    target = 0
    assert solution.searchInsert(nums, target) == 0
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
