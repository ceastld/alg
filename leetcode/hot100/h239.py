"""
LeetCode 239. Sliding Window Maximum

题目描述：
给你一个整数数组nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的k个数字。滑动窗口每次只向右移动一位。
返回每个滑动窗口中的最大值。

示例：
nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]

数据范围：
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- 1 <= k <= nums.length
"""

class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        from collections import deque
        
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            # 移除窗口外的元素
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # 移除比当前元素小的元素
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # 当窗口形成时，记录最大值
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
