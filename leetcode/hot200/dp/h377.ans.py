"""
377. 组合总和IV - 标准答案
"""
from typing import List


class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        """
        标准解法：动态规划（完全背包）
        
        解题思路：
        1. 问题转化为：在限制总金额的情况下，选择硬币的组合数（考虑顺序）
        2. 使用一维DP：dp[i] 表示金额为i时的组合数
        3. 状态转移方程：
           - dp[i] += dp[i - num]
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        # 使用一维数组优化空间
        dp = [0] * (target + 1)
        dp[0] = 1  # 金额为0时有一种组合（不选择任何数字）
        
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        
        return dp[target]
    
    def combinationSum4_2d(self, nums: List[int], target: int) -> int:
        """
        二维DP解法
        
        解题思路：
        1. dp[i][j] 表示前i个数字组成金额j的组合数
        2. 状态转移方程：
           - dp[i][j] = dp[i-1][j] + dp[i][j-nums[i-1]]
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target * len(nums))
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        n = len(nums)
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        
        # 初始化：金额为0时有一种组合
        for i in range(n + 1):
            dp[i][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if j >= nums[i-1]:
                    # 可以选择当前数字
                    dp[i][j] = dp[i-1][j] + dp[i][j-nums[i-1]]
                else:
                    # 不能选择当前数字
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][target]
    
    def combinationSum4_recursive(self, nums: List[int], target: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个金额的组合数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        memo = {}
        
        def dfs(remaining):
            if remaining in memo:
                return memo[remaining]
            
            if remaining == 0:
                return 1
            
            if remaining < 0:
                return 0
            
            result = 0
            for num in nums:
                if remaining >= num:
                    result += dfs(remaining - num)
            
            memo[remaining] = result
            return result
        
        return dfs(target)
    
    def combinationSum4_brute_force(self, nums: List[int], target: int) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的数字组合
        2. 计算每种组合的总和
        3. 统计等于target的组合数
        
        时间复杂度：O(target^len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        def dfs(remaining, current_combination):
            if remaining == 0:
                return 1
            
            if remaining < 0:
                return 0
            
            result = 0
            for num in nums:
                if remaining >= num:
                    result += dfs(remaining - num, current_combination + [num])
            
            return result
        
        return dfs(target, [])
    
    def combinationSum4_optimized(self, nums: List[int], target: int) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 从前往后遍历，避免重复使用
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        # 使用一维数组优化空间
        dp = [0] * (target + 1)
        dp[0] = 1  # 金额为0时有一种组合
        
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        
        return dp[target]
    
    def combinationSum4_alternative(self, nums: List[int], target: int) -> int:
        """
        替代解法：使用集合
        
        解题思路：
        1. 使用集合存储所有可能的金额
        2. 逐步更新集合
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        # 使用集合存储所有可能的金额
        possible_amounts = {0: 1}  # 初始状态：金额为0有1种组合
        
        for i in range(1, target + 1):
            new_amounts = {}
            for num in nums:
                if i >= num:
                    new_amounts[i] = new_amounts.get(i, 0) + possible_amounts.get(i - num, 0)
            
            # 更新集合
            for amt, count in new_amounts.items():
                possible_amounts[amt] = possible_amounts.get(amt, 0) + count
        
        return possible_amounts.get(target, 0)
    
    def combinationSum4_dfs(self, nums: List[int], target: int) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的数字组合
        2. 统计等于target的组合数
        
        时间复杂度：O(target^len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        def dfs(remaining):
            if remaining == 0:
                return 1
            
            if remaining < 0:
                return 0
            
            result = 0
            for num in nums:
                if remaining >= num:
                    result += dfs(remaining - num)
            
            return result
        
        return dfs(target)
    
    def combinationSum4_memo(self, nums: List[int], target: int) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        memo = {}
        
        def dfs(remaining):
            if remaining in memo:
                return memo[remaining]
            
            if remaining == 0:
                return 1
            
            if remaining < 0:
                return 0
            
            result = 0
            for num in nums:
                if remaining >= num:
                    result += dfs(remaining - num)
            
            memo[remaining] = result
            return result
        
        return dfs(target)
    
    def combinationSum4_greedy(self, nums: List[int], target: int) -> int:
        """
        贪心解法：优先选择小数字
        
        解题思路：
        1. 优先选择较小的数字
        2. 贪心地选择最优解
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        # 按数字从小到大排序
        sorted_nums = sorted(nums)
        
        def dfs(remaining):
            if remaining == 0:
                return 1
            
            if remaining < 0:
                return 0
            
            result = 0
            for num in sorted_nums:
                if remaining >= num:
                    result += dfs(remaining - num)
                else:
                    break  # 由于排序，后续数字都更大，可以提前退出
            
            return result
        
        return dfs(target)
    
    def combinationSum4_iterative(self, nums: List[int], target: int) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(target * len(nums))
        空间复杂度：O(target)
        """
        if target == 0:
            return 1
        
        if not nums:
            return 0
        
        # 使用栈模拟递归
        stack = [(target, 0)]  # (remaining, count)
        result = 0
        
        while stack:
            remaining, count = stack.pop()
            
            if remaining == 0:
                result += 1
                continue
            
            if remaining < 0:
                continue
            
            for num in nums:
                if remaining >= num:
                    stack.append((remaining - num, count + 1))
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,2,3]
    target = 4
    result = solution.combinationSum4(nums, target)
    expected = 7
    assert result == expected
    
    # 测试用例2
    nums = [9]
    target = 3
    result = solution.combinationSum4(nums, target)
    expected = 0
    assert result == expected
    
    # 测试用例3
    nums = [1,2,3]
    target = 1
    result = solution.combinationSum4(nums, target)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,2,3]
    target = 2
    result = solution.combinationSum4(nums, target)
    expected = 2
    assert result == expected
    
    # 测试用例5
    nums = [1,2,3]
    target = 3
    result = solution.combinationSum4(nums, target)
    expected = 4
    assert result == expected
    
    # 测试二维DP解法
    print("测试二维DP解法...")
    nums = [1,2,3]
    target = 4
    result_2d = solution.combinationSum4_2d(nums, target)
    assert result_2d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [1,2,3]
    target = 4
    result_rec = solution.combinationSum4_recursive(nums, target)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,2,3]
    target = 4
    result_opt = solution.combinationSum4_optimized(nums, target)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    nums = [1,2,3]
    target = 4
    result_alt = solution.combinationSum4_alternative(nums, target)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    nums = [1,2,3]
    target = 4
    result_dfs = solution.combinationSum4_dfs(nums, target)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    nums = [1,2,3]
    target = 4
    result_memo = solution.combinationSum4_memo(nums, target)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    nums = [1,2,3]
    target = 4
    result_greedy = solution.combinationSum4_greedy(nums, target)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    nums = [1,2,3]
    target = 4
    result_iter = solution.combinationSum4_iterative(nums, target)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
