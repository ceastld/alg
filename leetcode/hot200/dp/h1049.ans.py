"""
1049. 最后一块石头的重量II - 标准答案
"""
from typing import List


class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        """
        标准解法：动态规划（0-1背包）
        
        解题思路：
        1. 问题转化为：将石头分成两堆，使得两堆重量差最小
        2. 设总重量为sum，两堆重量分别为a和b，则a + b = sum
        3. 最小重量差为 |a - b| = |2a - sum|，当a接近sum/2时最小
        4. 使用0-1背包找到最接近sum/2的子集和
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not stones:
            return 0
        
        total_sum = sum(stones)
        target = total_sum // 2
        
        # 使用一维数组优化空间
        dp = [False] * (target + 1)
        dp[0] = True
        
        for stone in stones:
            # 从后往前遍历，避免重复使用
            for j in range(target, stone - 1, -1):
                dp[j] = dp[j] or dp[j - stone]
        
        # 找到最大的可达重量
        for j in range(target, -1, -1):
            if dp[j]:
                return total_sum - 2 * j
        
        return total_sum
    
    def lastStoneWeightII_2d(self, stones: List[int]) -> int:
        """
        二维DP解法
        
        解题思路：
        1. dp[i][j] 表示前i个石头是否能组成重量j
        2. 状态转移方程：
           - dp[i][j] = dp[i-1][j] or dp[i-1][j-stones[i-1]]
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not stones:
            return 0
        
        total_sum = sum(stones)
        target = total_sum // 2
        
        n = len(stones)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        
        # 初始化：重量为0总是可以组成
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if j < stones[i-1]:
                    # 当前石头重量大于目标和，不能选择
                    dp[i][j] = dp[i-1][j]
                else:
                    # 可以选择或不选择当前石头
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-stones[i-1]]
        
        # 找到最大的可达重量
        for j in range(target, -1, -1):
            if dp[n][j]:
                return total_sum - 2 * j
        
        return total_sum
    
    def lastStoneWeightII_recursive(self, stones: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置是否能组成目标和
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not stones:
            return 0
        
        total_sum = sum(stones)
        target = total_sum // 2
        
        memo = {}
        
        def dfs(i, remaining):
            if (i, remaining) in memo:
                return memo[(i, remaining)]
            
            if remaining == 0:
                return True
            
            if i >= len(stones) or remaining < 0:
                return False
            
            # 选择当前石头或不选择
            result = dfs(i + 1, remaining - stones[i]) or dfs(i + 1, remaining)
            memo[(i, remaining)] = result
            return result
        
        # 找到最大的可达重量
        for j in range(target, -1, -1):
            if dfs(0, j):
                return total_sum - 2 * j
        
        return total_sum
    
    def lastStoneWeightII_brute_force(self, stones: List[int]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子集
        2. 计算每种分割的最小重量差
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not stones:
            return 0
        
        total_sum = sum(stones)
        min_diff = float('inf')
        
        def dfs(i, sum1, sum2):
            nonlocal min_diff
            
            if i == len(stones):
                min_diff = min(min_diff, abs(sum1 - sum2))
                return
            
            # 将当前石头放入第一堆
            dfs(i + 1, sum1 + stones[i], sum2)
            # 将当前石头放入第二堆
            dfs(i + 1, sum1, sum2 + stones[i])
        
        dfs(0, 0, 0)
        return min_diff
    
    def lastStoneWeightII_optimized(self, stones: List[int]) -> int:
        """
        优化解法：位运算（参考canPartition方法）
        
        解题思路：
        1. 使用位运算表示所有可能的和
        2. 通过位运算快速计算
        3. 参考canPartition的位运算优化方法
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not stones:
            return 0
        
        total_sum = sum(stones)
        target = total_sum // 2
        
        # 使用位运算表示所有可能的和，从target开始
        dp = 1 << target
        
        for stone in stones:
            # 更新所有可能的和
            dp |= dp >> stone
            # 如果已经可以组成target，提前返回
            if dp & 1:
                return total_sum - 2 * target
        
        # 找到最大的可达重量
        for j in range(target, -1, -1):
            if dp & (1 << (target - j)):
                return total_sum - 2 * j
        
        return total_sum
    
    def lastStoneWeightII_alternative(self, stones: List[int]) -> int:
        """
        替代解法：使用集合
        
        解题思路：
        1. 使用集合存储所有可能的和
        2. 逐步更新集合
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not stones:
            return 0
        
        total_sum = sum(stones)
        target = total_sum // 2
        
        # 使用集合存储所有可能的和
        possible_sums = {0}
        
        for stone in stones:
            # 创建新的集合，避免重复使用
            new_sums = set()
            for s in possible_sums:
                new_sum = s + stone
                if new_sum <= target:
                    new_sums.add(new_sum)
            
            possible_sums.update(new_sums)
        
        # 找到最大的可达重量
        max_sum = 0
        for s in possible_sums:
            if s <= target:
                max_sum = max(max_sum, s)
        
        return total_sum - 2 * max_sum
    
    def lastStoneWeightII_greedy(self, stones: List[int]) -> int:
        """
        贪心解法：优先队列
        
        解题思路：
        1. 使用优先队列模拟粉碎过程
        2. 每次选择最大的两块石头粉碎
        3. 直到只剩一块或没有石头
        
        时间复杂度：O(n log n)
        空间复杂度：O(n)
        """
        if not stones:
            return 0
        
        import heapq
        
        # 使用最大堆
        heap = [-stone for stone in stones]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            # 取出最大的两块石头
            stone1 = -heapq.heappop(heap)
            stone2 = -heapq.heappop(heap)
            
            # 粉碎后剩余重量
            remaining = abs(stone1 - stone2)
            if remaining > 0:
                heapq.heappush(heap, -remaining)
        
        return -heap[0] if heap else 0


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    stones = [2,7,4,1,8,1]
    result = solution.lastStoneWeightII(stones)
    expected = 1
    assert result == expected
    
    # 测试用例2
    stones = [31,26,33,21,40]
    result = solution.lastStoneWeightII(stones)
    expected = 5
    assert result == expected
    
    # 测试用例3
    stones = [1,2]
    result = solution.lastStoneWeightII(stones)
    expected = 1
    assert result == expected
    
    # 测试用例4
    stones = [1,1,2,3,5,8,13,21,34,55]
    result = solution.lastStoneWeightII(stones)
    expected = 0
    assert result == expected
    
    # 测试用例5
    stones = [1]
    result = solution.lastStoneWeightII(stones)
    expected = 1
    assert result == expected
    
    # 测试二维DP解法
    print("测试二维DP解法...")
    stones = [2,7,4,1,8,1]
    result_2d = solution.lastStoneWeightII_2d(stones)
    assert result_2d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    stones = [2,7,4,1,8,1]
    result_rec = solution.lastStoneWeightII_recursive(stones)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    stones = [2,7,4,1,8,1]
    result_opt = solution.lastStoneWeightII_optimized(stones)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    stones = [2,7,4,1,8,1]
    result_alt = solution.lastStoneWeightII_alternative(stones)
    assert result_alt == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    stones = [2,7,4,1,8,1]
    result_greedy = solution.lastStoneWeightII_greedy(stones)
    assert result_greedy == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
