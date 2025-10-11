"""
35. 搜索插入位置 - 标准答案
"""
from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        """
        标准解法：二分查找
        
        解题思路：
        1. 使用二分查找找到target的位置
        2. 如果找到，返回索引
        3. 如果没找到，返回应该插入的位置（left指针的位置）
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return left


def main():
    """测试标准答案"""
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
