"""
LeetCode 494. Target Sum

题目描述：
给你一个整数数组nums和一个整数target。
向数组中的每个整数前添加'+'或'-'，然后串联起所有整数，可以构造一个表达式：
例如，nums = [2, 1]，可以在2之前添加'+'，在1之前添加'-'，然后串联起来得到表达式"+2-1"。
返回可以通过上述方法构造的、运算结果等于target的不同表达式的数目。

示例：
nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有5种方法让最终目标和为3。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

数据范围：
- 1 <= nums.length <= 20
- 0 <= nums[i] <= 1000
- 0 <= sum(nums[i]) <= 1000
- -1000 <= target <= 1000
"""

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
