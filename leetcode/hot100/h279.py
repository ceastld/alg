class Solution:
    def numSquares(self, n: int) -> int:
        # 方法1：动态规划
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        
        return dp[n]
    
    def numSquaresMath(self, n: int) -> int:
        # 方法2：数学解法（基于四平方定理）
        def is_square(num):
            root = int(num ** 0.5)
            return root * root == num
        
        # 四平方定理：任何正整数都可以表示为4个整数的平方和
        # 特殊情况：
        # 1. 如果n是完全平方数，返回1
        if is_square(n):
            return 1
        
        # 2. 如果n可以表示为4^a(8b+7)的形式，返回4
        temp = n
        while temp % 4 == 0:
            temp //= 4
        if temp % 8 == 7:
            return 4
        
        # 3. 检查是否可以表示为两个完全平方数的和
        for i in range(1, int(n ** 0.5) + 1):
            if is_square(n - i * i):
                return 2
        
        # 4. 其他情况返回3
        return 3
