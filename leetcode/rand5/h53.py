from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        s1 = 0
        s2 = 0
        max_sum = float('-inf')
        for num in nums:
            s1 = min(s1,s2)
            s2 += num 
            if s2 - s1 > max_sum:
                max_sum = s2 - s1
        return max_sum