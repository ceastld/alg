"""
941. 有效的山脉数组 - 标准答案
"""
from typing import List


class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        """
        标准解法：双指针
        
        解题思路：
        1. 从左边开始，找到第一个不满足递增的位置
        2. 从右边开始，找到第一个不满足递增的位置
        3. 如果两个位置相同且不在边界，则是有效山脉
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(arr)
        if n < 3:
            return False
        
        left = 0
        while left < n - 1 and arr[left] < arr[left + 1]:
            left += 1
        
        right = n - 1
        while right > 0 and arr[right] < arr[right - 1]:
            right -= 1
        
        return left == right and left != 0 and right != n - 1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    arr = [2, 1]
    assert solution.validMountainArray(arr) == False
    
    # 测试用例2
    arr = [3, 5, 5]
    assert solution.validMountainArray(arr) == False
    
    # 测试用例3
    arr = [0, 3, 2, 1]
    assert solution.validMountainArray(arr) == True
    
    # 测试用例4
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert solution.validMountainArray(arr) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
