"""
322. 零钱兑换 - 标准答案
"""
from typing import List


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示凑成金额i所需的最少硬币数
        2. 状态转移方程：dp[i] = min(dp[i-coin] + 1) for coin in coins
        3. 边界条件：dp[0] = 0
        4. 初始化：其他金额设为无穷大
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 0
        
        # 初始化DP数组
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        # 遍历所有金额
        for i in range(1, amount + 1):
            # 尝试每种硬币
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def coinChange_optimized(self, coins: List[int], amount: int) -> int:
        """
        优化解法：先排序硬币
        
        解题思路：
        1. 先对硬币面额排序
        2. 如果当前金额已经大于等于某个硬币面额，可以提前终止
        3. 减少不必要的计算
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 0
        
        coins.sort()
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin > i:
                    break  # 硬币面额大于当前金额，提前终止
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def coinChange_recursive(self, coins: List[int], amount: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 0
        
        memo = {}
        
        def dfs(remaining):
            if remaining in memo:
                return memo[remaining]
            if remaining < 0:
                return float('inf')
            if remaining == 0:
                return 0
            
            min_coins = float('inf')
            for coin in coins:
                result = dfs(remaining - coin)
                if result != float('inf'):
                    min_coins = min(min_coins, result + 1)
            
            memo[remaining] = min_coins
            return min_coins
        
        result = dfs(amount)
        return result if result != float('inf') else -1
    
    def coinChange_bfs(self, coins: List[int], amount: int) -> int:
        """
        BFS解法：广度优先搜索
        
        解题思路：
        1. 将问题转化为图的最短路径问题
        2. 每个金额是一个节点，硬币是边
        3. 使用BFS找到从0到amount的最短路径
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 0
        
        from collections import deque
        
        queue = deque([0])
        visited = {0}
        level = 0
        
        while queue:
            size = len(queue)
            for _ in range(size):
                current = queue.popleft()
                
                if current == amount:
                    return level
                
                for coin in coins:
                    next_amount = current + coin
                    if next_amount <= amount and next_amount not in visited:
                        visited.add(next_amount)
                        queue.append(next_amount)
            
            level += 1
        
        return -1
    
    def coinChange_greedy_fail(self, coins: List[int], amount: int) -> int:
        """
        贪心解法（注意：这个解法是错误的！）
        
        解题思路：
        1. 每次都选择面额最大的硬币
        2. 这种方法在某些情况下会得到错误结果
        
        时间复杂度：O(len(coins))
        空间复杂度：O(1)
        
        注意：这个解法是错误的，仅用于演示贪心算法的局限性
        """
        if amount == 0:
            return 0
        
        coins.sort(reverse=True)
        count = 0
        remaining = amount
        
        for coin in coins:
            if coin <= remaining:
                count += remaining // coin
                remaining %= coin
        
        return count if remaining == 0 else -1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    coins = [1, 3, 4]
    amount = 6
    result = solution.coinChange(coins, amount)
    expected = 2
    assert result == expected
    
    # 测试用例2
    coins = [2]
    amount = 3
    result = solution.coinChange(coins, amount)
    expected = -1
    assert result == expected
    
    # 测试用例3
    coins = [1]
    amount = 0
    result = solution.coinChange(coins, amount)
    expected = 0
    assert result == expected
    
    # 测试用例4
    coins = [1, 2, 5]
    amount = 11
    result = solution.coinChange(coins, amount)
    expected = 3
    assert result == expected
    
    # 测试用例5
    coins = [2, 5, 10, 1]
    amount = 27
    result = solution.coinChange(coins, amount)
    expected = 4
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    coins = [1, 3, 4]
    amount = 6
    result_opt = solution.coinChange_optimized(coins, amount)
    expected_opt = 2
    assert result_opt == expected_opt
    
    # 测试递归解法
    print("测试递归解法...")
    coins = [1, 3, 4]
    amount = 6
    result_rec = solution.coinChange_recursive(coins, amount)
    expected_rec = 2
    assert result_rec == expected_rec
    
    # 测试BFS解法
    print("测试BFS解法...")
    coins = [1, 3, 4]
    amount = 6
    result_bfs = solution.coinChange_bfs(coins, amount)
    expected_bfs = 2
    assert result_bfs == expected_bfs
    
    # 测试贪心解法（注意：这个解法是错误的）
    print("测试贪心解法（错误示例）...")
    coins = [1, 3, 4]
    amount = 6
    result_greedy = solution.coinChange_greedy_fail(coins, amount)
    # 贪心解法会得到错误结果，这里仅用于演示
    print(f"贪心解法结果: {result_greedy} (错误)")
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
