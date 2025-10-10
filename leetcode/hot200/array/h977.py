"""
977. 有序数组的平方
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，
要求也按 非递减顺序 排序。

题目链接：https://leetcode.cn/problems/squares-of-a-sorted-array/

示例 1:
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]

示例 2:
输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]

提示：
- 1 <= nums.length <= 104
- -104 <= nums[i] <= 104
- nums 已按 非递减顺序 排序
"""
from typing import List


class Solution:
    """
    977. 有序数组的平方
    双指针经典应用
    """
    
    def sortedSquares(self, nums: List[int]) -> List[int]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        nums = [num * num for num in nums]
        nums.sort()
        return nums


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.sortedSquares([-4, -1, 0, 3, 10]) == [0, 1, 9, 16, 100]
    
    # 测试用例2
    assert solution.sortedSquares([-7, -3, 2, 3, 11]) == [4, 9, 9, 49, 121]
    
    # 测试用例3
    assert solution.sortedSquares([-5, -3, -2, -1]) == [1, 4, 9, 25]
    
    # 测试用例4
    assert solution.sortedSquares([0, 1, 2, 3, 4]) == [0, 1, 4, 9, 16]
    
    # 测试用例5
    assert solution.sortedSquares([-1]) == [1]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
