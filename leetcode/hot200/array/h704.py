"""
704. 二分查找
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target ，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

题目链接：https://leetcode.cn/problems/binary-search/

示例 1:
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4

示例 2:
输入: nums = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不存在 nums 中因此返回 -1

提示：
- 你可以假设 nums 中的所有元素是不重复的
- n 将在 [1, 10000]之间
- nums 的每个元素都将在 [-9999, 9999]之间
"""
from typing import List


class Solution:
    """
    704. 二分查找
    经典二分查找模板题
    """
    
    def search(self, nums: List[int], target: int) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.search([-1, 0, 3, 5, 9, 12], 9) == 4
    
    # 测试用例2
    assert solution.search([-1, 0, 3, 5, 9, 12], 2) == -1
    
    # 测试用例3
    assert solution.search([5], 5) == 0
    
    # 测试用例4
    assert solution.search([5], -5) == -1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
