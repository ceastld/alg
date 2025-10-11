"""
34. 在排序数组中查找元素的第一个和最后一个位置 - 标准答案
"""
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        标准解法：二分查找
        
        解题思路：
        1. 使用二分查找找到第一个等于target的位置
        2. 使用二分查找找到最后一个等于target的位置
        3. 如果没找到，返回[-1, -1]
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        def findFirst(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    if mid == 0 or nums[mid - 1] != target:
                        return mid
                    right = mid - 1
            return -1
        
        def findLast(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    if mid == len(nums) - 1 or nums[mid + 1] != target:
                        return mid
                    left = mid + 1
            return -1
        
        if not nums:
            return [-1, -1]
        
        first = findFirst(nums, target)
        if first == -1:
            return [-1, -1]
        
        last = findLast(nums, target)
        return [first, last]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [5, 7, 7, 8, 8, 10]
    target = 8
    assert solution.searchRange(nums, target) == [3, 4]
    
    # 测试用例2
    nums = [5, 7, 7, 8, 8, 10]
    target = 6
    assert solution.searchRange(nums, target) == [-1, -1]
    
    # 测试用例3
    nums = []
    target = 0
    assert solution.searchRange(nums, target) == [-1, -1]
    
    # 测试用例4
    nums = [1]
    target = 1
    assert solution.searchRange(nums, target) == [0, 0]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
