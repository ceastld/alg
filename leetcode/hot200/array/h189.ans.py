"""
189. 旋转数组 - 标准答案
"""
from typing import List


class Solution:
    """
    189. 旋转数组 - 标准解法
    """
    
    def rotate(self, nums: List[int], k: int) -> None:
        """
        标准解法：三次反转法
        
        解题思路：
        1. 先对k取模，因为旋转n次等于不旋转
        2. 使用三次反转：先反转整个数组，再分别反转前k个和后n-k个
        3. 这样就能实现向右旋转k个位置的效果
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        k = k % n  # 处理k大于n的情况
        
        # 反转整个数组
        self.reverse(nums, 0, n - 1)
        # 反转前k个元素
        self.reverse(nums, 0, k - 1)
        # 反转后n-k个元素
        self.reverse(nums, k, n - 1)
    
    def reverse(self, nums: List[int], start: int, end: int) -> None:
        """反转数组的指定区间"""
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums_1 = [1, 2, 3, 4, 5, 6, 7]
    solution.rotate(nums_1, 3)
    assert nums_1 == [5, 6, 7, 1, 2, 3, 4]
    
    # 测试用例2
    nums_2 = [-1, -100, 3, 99]
    solution.rotate(nums_2, 2)
    assert nums_2 == [3, 99, -1, -100]
    
    # 测试用例3
    nums_3 = [1, 2]
    solution.rotate(nums_3, 1)
    assert nums_3 == [2, 1]
    
    # 测试用例4
    nums_4 = [1, 2, 3, 4, 5]
    solution.rotate(nums_4, 0)
    assert nums_4 == [1, 2, 3, 4, 5]
    
    # 测试用例5
    nums_5 = [1, 2, 3, 4, 5, 6]
    solution.rotate(nums_5, 2)
    assert nums_5 == [5, 6, 1, 2, 3, 4]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
