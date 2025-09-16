class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        from collections import defaultdict
        
        count = 0
        prefix_sum = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1  # 前缀和为0出现1次
        
        for num in nums:
            prefix_sum += num
            # 如果存在前缀和为 prefix_sum - k，则存在子数组和为k
            if prefix_sum - k in sum_count:
                count += sum_count[prefix_sum - k]
            
            # 记录当前前缀和的出现次数
            sum_count[prefix_sum] += 1
        
        return count
