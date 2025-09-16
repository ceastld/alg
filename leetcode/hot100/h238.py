from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 非零元素乘积，
        # 零元素个数，如果 >=0
        prod = 1
        zero_count = 0
        for n in nums:
            if n != 0:
                prod *= n
            else:
                zero_count += 1
        if zero_count >= 2:
            return [0] * len(nums)
        elif zero_count == 1:
            return [prod if n == 0 else 0 for n in nums]
        else:
            return [prod if n == 0 else prod // n for n in nums]