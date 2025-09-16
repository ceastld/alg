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
