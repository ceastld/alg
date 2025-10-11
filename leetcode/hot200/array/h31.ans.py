"""
31. 下一个排列 - 标准答案
"""
from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        标准解法：双指针 + 原地修改
        
        解题思路：
        1. 从右往左找到第一个降序的位置i（nums[i] < nums[i+1]）
        2. 从右往左找到第一个大于nums[i]的位置j
        3. 交换nums[i]和nums[j]
        4. 将i+1到末尾的部分反转
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        if n <= 1:
            return
        
        # 步骤1：从右往左找到第一个降序的位置
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            # 步骤2：从右往左找到第一个大于nums[i]的位置
            j = n - 1
            while j > i and nums[j] <= nums[i]:
                j -= 1
            
            # 步骤3：交换nums[i]和nums[j]
            nums[i], nums[j] = nums[j], nums[i]
        
        # 步骤4：将i+1到末尾的部分反转
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 2, 3]
    solution.nextPermutation(nums)
    assert nums == [1, 3, 2]
    
    # 测试用例2
    nums = [3, 2, 1]
    solution.nextPermutation(nums)
    assert nums == [1, 2, 3]
    
    # 测试用例3
    nums = [1, 1, 5]
    solution.nextPermutation(nums)
    assert nums == [1, 5, 1]
    
    # 测试用例4
    nums = [1, 3, 2]
    solution.nextPermutation(nums)
    assert nums == [2, 1, 3]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
