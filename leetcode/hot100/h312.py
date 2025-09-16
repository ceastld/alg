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
