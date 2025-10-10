"""
977. 有序数组的平方 - 标准答案
"""
from typing import List


class Solution:
    """
    977. 有序数组的平方 - 标准解法
    """
    
    def sortedSquares(self, nums: List[int]) -> List[int]:
        """
        标准解法：双指针法
        
        解题思路：
        1. 由于原数组有序，平方后最大值在两端
        2. 使用双指针从两端向中间遍历
        3. 比较两端的平方值，将较大的放入结果数组末尾
        4. 重复直到所有元素处理完毕
        
        时间复杂度：O(n)
        空间复杂度：O(1) 不考虑输出数组
        """
        n = len(nums)
        result = [0] * n
        left, right = 0, n - 1
        index = n - 1
        
        while left <= right:
            left_square = nums[left] * nums[left]
            right_square = nums[right] * nums[right]
            
            if left_square > right_square:
                result[index] = left_square
                left += 1
            else:
                result[index] = right_square
                right -= 1
            index -= 1
        
        return result


def main():
    """测试标准答案"""
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
