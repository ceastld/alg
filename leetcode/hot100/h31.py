"""
LeetCode 31. Next Permutation

题目描述：
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

示例：
nums = [1,2,3]
输出：[1,3,2]

数据范围：
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 100
"""

class Solution:
    def nextPermutation(self, nums: list[int]) -> None:
        # 1. 从右往左找到第一个下降的位置
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            # 2. 从右往左找到第一个大于nums[i]的位置
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            
            # 3. 交换nums[i]和nums[j]
            nums[i], nums[j] = nums[j], nums[i]
        
        # 4. 反转i+1到末尾的部分
        nums[i + 1:] = nums[i + 1:][::-1]
