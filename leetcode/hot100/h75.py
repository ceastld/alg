"""
LeetCode 75. Sort Colors

题目描述：
给定一个包含红色、白色和蓝色、共n个元素的数组nums，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
我们使用整数0、1和2分别表示红色、白色和蓝色。
必须在不使用库内置的sort函数的情况下解决这个问题。

示例：
nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]

数据范围：
- n == nums.length
- 1 <= n <= 300
- nums[i]为0、1或2
"""

class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 三指针法：left指向0的右边界，right指向2的左边界，curr是当前指针
        left = curr = 0
        right = len(nums) - 1
        
        while curr <= right:
            if nums[curr] == 0:
                # 交换到左边
                nums[left], nums[curr] = nums[curr], nums[left]
                left += 1
                curr += 1
            elif nums[curr] == 2:
                # 交换到右边
                nums[curr], nums[right] = nums[right], nums[curr]
                right -= 1
                # 注意：这里curr不增加，因为交换过来的元素还需要检查
            else:
                # nums[curr] == 1，直接跳过
                curr += 1
