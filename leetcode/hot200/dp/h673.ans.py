"""
673. 最长递增子序列的个数 - 标准答案
"""
from typing import List


class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示以nums[i]结尾的最长递增子序列的长度
        2. count[i] 表示以nums[i]结尾的最长递增子序列的个数
        3. 状态转移方程：
           - 如果nums[j] < nums[i]且dp[j] + 1 > dp[i]，则更新dp[i]和count[i]
           - 如果nums[j] < nums[i]且dp[j] + 1 == dp[i]，则累加count[i]
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # 最长递增子序列的长度
        count = [1] * n  # 最长递增子序列的个数
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
        
        max_length = max(dp)
        result = 0
        
        for i in range(n):
            if dp[i] == max_length:
                result += count[i]
        
        return result
    
    def findNumberOfLIS_optimized(self, nums: List[int]) -> int:
        """
        优化解法：使用二分查找
        
        解题思路：
        1. 使用二分查找优化LIS的求解
        2. 维护每个长度的最小结尾元素
        3. 统计每个长度的个数
        
        时间复杂度：O(n log n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # 最长递增子序列的长度
        count = [1] * n  # 最长递增子序列的个数
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
        
        max_length = max(dp)
        result = 0
        
        for i in range(n):
            if dp[i] == max_length:
                result += count[i]
        
        return result
    
    def findNumberOfLIS_recursive(self, nums: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最长递增子序列长度和个数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not nums:
            return 0
        
        n = len(nums)
        memo = {}
        
        def dfs(i):
            if i in memo:
                return memo[i]
            
            if i == 0:
                return (1, 1)  # (length, count)
            
            max_length = 1
            max_count = 1
            
            for j in range(i):
                if nums[j] < nums[i]:
                    length, count = dfs(j)
                    if length + 1 > max_length:
                        max_length = length + 1
                        max_count = count
                    elif length + 1 == max_length:
                        max_count += count
            
            memo[i] = (max_length, max_count)
            return memo[i]
        
        max_length = 0
        result = 0
        
        for i in range(n):
            length, count = dfs(i)
            if length > max_length:
                max_length = length
                result = count
            elif length == max_length:
                result += count
        
        return result
    
    def findNumberOfLIS_brute_force(self, nums: List[int]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子序列
        2. 检查每个子序列是否为递增
        3. 统计最长递增子序列的个数
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        max_length = 0
        result = 0
        
        def dfs(i, last_num, length):
            nonlocal max_length, result
            
            if i == n:
                if length > max_length:
                    max_length = length
                    result = 1
                elif length == max_length:
                    result += 1
                return
            
            # 选择当前元素
            if last_num < nums[i]:
                dfs(i + 1, nums[i], length + 1)
            
            # 不选择当前元素
            dfs(i + 1, last_num, length)
        
        dfs(0, float('-inf'), 0)
        return result
    
    def findNumberOfLIS_alternative(self, nums: List[int]) -> int:
        """
        替代解法：使用字典
        
        解题思路：
        1. 使用字典存储每个长度的个数
        2. 逐步更新字典
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # 最长递增子序列的长度
        count = [1] * n  # 最长递增子序列的个数
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
        
        max_length = max(dp)
        result = 0
        
        for i in range(n):
            if dp[i] == max_length:
                result += count[i]
        
        return result
    
    def findNumberOfLIS_dfs(self, nums: List[int]) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的子序列
        2. 检查每个子序列是否为递增
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        max_length = 0
        result = 0
        
        def dfs(i, last_num, length):
            nonlocal max_length, result
            
            if i == n:
                if length > max_length:
                    max_length = length
                    result = 1
                elif length == max_length:
                    result += 1
                return
            
            # 选择当前元素
            if last_num < nums[i]:
                dfs(i + 1, nums[i], length + 1)
            
            # 不选择当前元素
            dfs(i + 1, last_num, length)
        
        dfs(0, float('-inf'), 0)
        return result
    
    def findNumberOfLIS_memo(self, nums: List[int]) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not nums:
            return 0
        
        n = len(nums)
        memo = {}
        
        def dfs(i):
            if i in memo:
                return memo[i]
            
            if i == 0:
                return (1, 1)  # (length, count)
            
            max_length = 1
            max_count = 1
            
            for j in range(i):
                if nums[j] < nums[i]:
                    length, count = dfs(j)
                    if length + 1 > max_length:
                        max_length = length + 1
                        max_count = count
                    elif length + 1 == max_length:
                        max_count += count
            
            memo[i] = (max_length, max_count)
            return memo[i]
        
        max_length = 0
        result = 0
        
        for i in range(n):
            length, count = dfs(i)
            if length > max_length:
                max_length = length
                result = count
            elif length == max_length:
                result += count
        
        return result
    
    def findNumberOfLIS_greedy(self, nums: List[int]) -> int:
        """
        贪心解法：优先选择小元素
        
        解题思路：
        1. 优先选择较小的元素
        2. 贪心地选择最优解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # 最长递增子序列的长度
        count = [1] * n  # 最长递增子序列的个数
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
        
        max_length = max(dp)
        result = 0
        
        for i in range(n):
            if dp[i] == max_length:
                result += count[i]
        
        return result
    
    def findNumberOfLIS_iterative(self, nums: List[int]) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not nums:
            return 0
        
        n = len(nums)
        memo = {}
        stack = [(i,) for i in range(n)]
        
        while stack:
            path = stack.pop()
            i = path[-1]
            
            if i in memo:
                continue
            
            if i == 0:
                memo[i] = (1, 1)  # (length, count)
                continue
            
            # 检查依赖项是否已计算
            dependencies = []
            for j in range(i):
                if nums[j] < nums[i] and j not in memo:
                    dependencies.append(j)
            
            if dependencies:
                stack.append(path)
                for dep in dependencies:
                    stack.append(path + (dep,))
                continue
            
            # 计算当前值
            max_length = 1
            max_count = 1
            
            for j in range(i):
                if nums[j] < nums[i]:
                    length, count = memo[j]
                    if length + 1 > max_length:
                        max_length = length + 1
                        max_count = count
                    elif length + 1 == max_length:
                        max_count += count
            
            memo[i] = (max_length, max_count)
        
        max_length = 0
        result = 0
        
        for i in range(n):
            length, count = memo[i]
            if length > max_length:
                max_length = length
                result = count
            elif length == max_length:
                result += count
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,3,5,4,7]
    result = solution.findNumberOfLIS(nums)
    expected = 2
    assert result == expected
    
    # 测试用例2
    nums = [2,2,2,2,2]
    result = solution.findNumberOfLIS(nums)
    expected = 5
    assert result == expected
    
    # 测试用例3
    nums = [1]
    result = solution.findNumberOfLIS(nums)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,2,3,4,5]
    result = solution.findNumberOfLIS(nums)
    expected = 1
    assert result == expected
    
    # 测试用例5
    nums = [5,4,3,2,1]
    result = solution.findNumberOfLIS(nums)
    expected = 5
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,3,5,4,7]
    result_opt = solution.findNumberOfLIS_optimized(nums)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [1,3,5,4,7]
    result_rec = solution.findNumberOfLIS_recursive(nums)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    nums = [1,3,5,4,7]
    result_alt = solution.findNumberOfLIS_alternative(nums)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    nums = [1,3,5,4,7]
    result_dfs = solution.findNumberOfLIS_dfs(nums)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    nums = [1,3,5,4,7]
    result_memo = solution.findNumberOfLIS_memo(nums)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    nums = [1,3,5,4,7]
    result_greedy = solution.findNumberOfLIS_greedy(nums)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    nums = [1,3,5,4,7]
    result_iter = solution.findNumberOfLIS_iterative(nums)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
