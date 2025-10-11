"""
724. 寻找数组的中心索引 - 标准答案
"""
from typing import List


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        """
        标准解法：前缀和
        
        解题思路：
        1. 计算数组的总和
        2. 遍历数组，维护左侧和
        3. 如果左侧和等于右侧和（总和-左侧和-当前元素），返回当前索引
        4. 更新左侧和
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        total_sum = sum(nums)
        left_sum = 0
        
        for i, num in enumerate(nums):
            if left_sum == total_sum - left_sum - num:
                return i
            left_sum += num
        
        return -1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 7, 3, 6, 5, 6]
    assert solution.pivotIndex(nums) == 3
    
    # 测试用例2
    nums = [1, 2, 3]
    assert solution.pivotIndex(nums) == -1
    
    # 测试用例3
    nums = [2, 1, -1]
    assert solution.pivotIndex(nums) == 0
    
    # 测试用例4
    nums = [1, 2, 3, 4, 5]
    assert solution.pivotIndex(nums) == -1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
