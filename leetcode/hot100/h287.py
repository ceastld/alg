"""
LeetCode 287. Find the Duplicate Number

题目描述：
给定一个包含n+1个整数的数组nums，其数字都在[1, n]范围内（包括1和n），可知至少存在一个重复的整数。
假设nums只有一个重复的整数，返回这个重复的数。
你设计的解决方案必须不修改数组nums且只用常量级O(1)的额外空间。

示例：
nums = [1,3,4,2,2]
输出：2

数据范围：
- 1 <= n <= 10^5
- nums.length == n + 1
- 1 <= nums[i] <= n
- nums中只有一个整数出现两次或多次，其余整数均只出现一次
"""

class Solution:
    def findDuplicate(self, nums: list[int]) -> int:
        # 快慢指针找环
        slow = fast = nums[0]
        
        # 第一阶段：找到相遇点
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # 第二阶段：找到环的入口
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        
        return slow
