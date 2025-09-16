"""
LeetCode 312. Burst Balloons

题目描述：
有n个气球，编号为0到n-1，每个气球上都标有一个数字，这些数字存在数组nums中。
现在要求你戳破所有的气球。戳破第i个气球，你可以获得nums[i-1] * nums[i] * nums[i+1]枚硬币。这里的i-1和i+1代表和i相邻的两个气球的序号。如果i-1或i+1超出了数组的边界，那么就当它是一个数字为1的气球。
求所能获得硬币的最大数量。

示例：
nums = [3,1,5,8]
输出：167
解释：nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins = 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167

数据范围：
- n == nums.length
- 1 <= n <= 300
- 0 <= nums[i] <= 100
"""

class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        # 添加边界气球
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        # dp[i][j] 表示戳破区间(i,j)内所有气球的最大得分
        dp = [[0] * n for _ in range(n)]
        
        # 从小区间到大区间
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], 
                                  dp[i][k] + dp[k][j] + balloons[i] * balloons[k] * balloons[j])
        
        return dp[0][n - 1]
