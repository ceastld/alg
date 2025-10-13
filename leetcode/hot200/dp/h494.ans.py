"""
494. 目标和 - 标准答案
"""
from typing import List


class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        标准解法：动态规划（0-1背包）
        
        解题思路：
        1. 设正数集合为P，负数集合为N，则P - N = target
        2. 又因为P + N = sum(nums)，所以P = (target + sum) / 2
        3. 问题转化为：找到子集P，使得P的和等于(target + sum) / 2
        4. 使用0-1背包问题的解法
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not nums:
            return 0
        
        total_sum = sum(nums)
        
        # 如果target的绝对值大于总和，不可能
        if abs(target) > total_sum:
            return 0
        
        # 如果target和sum的奇偶性不同，不可能
        if (target + total_sum) % 2 != 0:
            return 0
        
        target_sum = (target + total_sum) // 2
        
        # 如果target_sum为负数，不可能
        if target_sum < 0:
            return 0
        
        # 使用一维数组优化空间
        dp = [0] * (target_sum + 1)
        dp[0] = 1  # 空子集的方案数为1
        
        for num in nums:
            # 从后往前遍历，避免重复使用
            for j in range(target_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[target_sum]
    
    def findTargetSumWays_2d(self, nums: List[int], target: int) -> int:
        """
        二维DP解法
        
        解题思路：
        1. dp[i][j] 表示前i个元素组成和为j的方案数
        2. 状态转移方程：
           - dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not nums:
            return 0
        
        total_sum = sum(nums)
        
        if abs(target) > total_sum:
            return 0
        
        if (target + total_sum) % 2 != 0:
            return 0
        
        target_sum = (target + total_sum) // 2
        
        if target_sum < 0:
            return 0
        
        n = len(nums)
        dp = [[0] * (target_sum + 1) for _ in range(n + 1)]
        
        # 初始化：空子集的方案数为1
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(target_sum + 1):
                if j < nums[i-1]:
                    # 当前元素大于目标和，不能选择
                    dp[i][j] = dp[i-1][j]
                else:
                    # 可以选择或不选择当前元素
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
        
        return dp[n][target_sum]
    
    def findTargetSumWays_recursive(self, nums: List[int], target: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的可能方案数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not nums:
            return 0
        
        total_sum = sum(nums)
        
        if abs(target) > total_sum:
            return 0
        
        if (target + total_sum) % 2 != 0:
            return 0
        
        target_sum = (target + total_sum) // 2
        
        if target_sum < 0:
            return 0
        
        memo = {}
        
        def dfs(i, remaining):
            if (i, remaining) in memo:
                return memo[(i, remaining)]
            
            if remaining == 0:
                return 1
            
            if i >= len(nums) or remaining < 0:
                return 0
            
            # 选择当前元素或不选择
            result = dfs(i + 1, remaining - nums[i]) + dfs(i + 1, remaining)
            memo[(i, remaining)] = result
            return result
        
        return dfs(0, target_sum)
    
    def findTargetSumWays_brute_force(self, nums: List[int], target: int) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的符号组合
        2. 计算每种组合的和
        3. 统计等于target的方案数
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        def dfs(i, current_sum):
            if i == len(nums):
                return 1 if current_sum == target else 0
            
            # 选择正号或负号
            return dfs(i + 1, current_sum + nums[i]) + dfs(i + 1, current_sum - nums[i])
        
        return dfs(0, 0)
    
    def findTargetSumWays_optimized(self, nums: List[int], target: int) -> int:
        """
        优化解法：位运算
        
        解题思路：
        1. 使用位运算表示所有可能的和
        2. 通过位运算快速计算
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not nums:
            return 0
        
        total_sum = sum(nums)
        
        if abs(target) > total_sum:
            return 0
        
        if (target + total_sum) % 2 != 0:
            return 0
        
        target_sum = (target + total_sum) // 2
        
        if target_sum < 0:
            return 0
        
        # 使用位运算表示所有可能的和
        possible_sums = 1  # 初始状态：和为0是可能的
        
        for num in nums:
            # 更新所有可能的和
            possible_sums |= (possible_sums << num)
        
        # 统计方案数
        count = 0
        for j in range(target_sum + 1):
            if possible_sums & (1 << j):
                count += 1
        
        return count
    
    def findTargetSumWays_alternative(self, nums: List[int], target: int) -> int:
        """
        替代解法：使用集合
        
        解题思路：
        1. 使用集合存储所有可能的和
        2. 逐步更新集合
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not nums:
            return 0
        
        total_sum = sum(nums)
        
        if abs(target) > total_sum:
            return 0
        
        if (target + total_sum) % 2 != 0:
            return 0
        
        target_sum = (target + total_sum) // 2
        
        if target_sum < 0:
            return 0
        
        # 使用集合存储所有可能的和
        possible_sums = {0: 1}  # 初始状态：和为0的方案数为1
        
        for num in nums:
            # 创建新的集合，避免重复使用
            new_sums = {}
            for s, count in possible_sums.items():
                new_sum = s + num
                if new_sum <= target_sum:
                    new_sums[new_sum] = new_sums.get(new_sum, 0) + count
            
            # 更新集合
            for s, count in new_sums.items():
                possible_sums[s] = possible_sums.get(s, 0) + count
        
        return possible_sums.get(target_sum, 0)
    
    def findTargetSumWays_dfs(self, nums: List[int], target: int) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的符号组合
        2. 统计等于target的方案数
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        def dfs(i, current_sum):
            if i == len(nums):
                return 1 if current_sum == target else 0
            
            # 选择正号或负号
            return dfs(i + 1, current_sum + nums[i]) + dfs(i + 1, current_sum - nums[i])
        
        return dfs(0, 0)
    
    def findTargetSumWays_memo(self, nums: List[int], target: int) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not nums:
            return 0
        
        memo = {}
        
        def dfs(i, current_sum):
            if (i, current_sum) in memo:
                return memo[(i, current_sum)]
            
            if i == len(nums):
                return 1 if current_sum == target else 0
            
            # 选择正号或负号
            result = dfs(i + 1, current_sum + nums[i]) + dfs(i + 1, current_sum - nums[i])
            memo[(i, current_sum)] = result
            return result
        
        return dfs(0, 0)


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,1,1,1,1]
    target = 3
    result = solution.findTargetSumWays(nums, target)
    expected = 5
    assert result == expected
    
    # 测试用例2
    nums = [1]
    target = 1
    result = solution.findTargetSumWays(nums, target)
    expected = 1
    assert result == expected
    
    # 测试用例3
    nums = [1,1,1,1,1]
    target = 5
    result = solution.findTargetSumWays(nums, target)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,1,1,1,1]
    target = 0
    result = solution.findTargetSumWays(nums, target)
    expected = 0
    assert result == expected
    
    # 测试用例5
    nums = [1,2,3,4,5]
    target = 3
    result = solution.findTargetSumWays(nums, target)
    expected = 3
    assert result == expected
    
    # 测试二维DP解法
    print("测试二维DP解法...")
    nums = [1,1,1,1,1]
    target = 3
    result_2d = solution.findTargetSumWays_2d(nums, target)
    assert result_2d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [1,1,1,1,1]
    target = 3
    result_rec = solution.findTargetSumWays_recursive(nums, target)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,1,1,1,1]
    target = 3
    result_opt = solution.findTargetSumWays_optimized(nums, target)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    nums = [1,1,1,1,1]
    target = 3
    result_alt = solution.findTargetSumWays_alternative(nums, target)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    nums = [1,1,1,1,1]
    target = 3
    result_dfs = solution.findTargetSumWays_dfs(nums, target)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    nums = [1,1,1,1,1]
    target = 3
    result_memo = solution.findTargetSumWays_memo(nums, target)
    assert result_memo == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
