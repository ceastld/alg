class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        if (total + target) % 2 == 1 or total < abs(target):
            return 0
        
        # 转换为背包问题：找到和为 (total + target) // 2 的子集数量
        subset_sum = (total + target) // 2
        
        # dp[i] 表示和为 i 的子集数量
        dp = [0] * (subset_sum + 1)
        dp[0] = 1
        
        for num in nums:
            for i in range(subset_sum, num - 1, -1):
                dp[i] += dp[i - num]
        
        return dp[subset_sum]
