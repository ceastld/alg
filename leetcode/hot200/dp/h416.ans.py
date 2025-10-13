"""
416. 分割等和子集 - 标准答案
"""
from typing import List


class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        """
        标准解法：动态规划（0-1背包）
        
        解题思路：
        1. 如果数组和是奇数，不可能分割成两个相等的子集
        2. 如果数组和是偶数，问题转化为：是否存在子集，其和等于数组和的一半
        3. 使用0-1背包问题的解法
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not nums:
            return False
        
        total_sum = sum(nums)
        
        # 如果和是奇数，不可能分割成两个相等的子集
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        # 如果最大元素大于目标值，不可能
        if max(nums) > target:
            return False
        
        # 使用一维数组优化空间
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            # 从后往前遍历，避免重复使用
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[target]
    
    def canPartition_2d(self, nums: List[int]) -> bool:
        """
        二维DP解法
        
        解题思路：
        1. dp[i][j] 表示前i个元素是否能组成和为j的子集
        2. 状态转移方程：
           - dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not nums:
            return False
        
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        if max(nums) > target:
            return False
        
        n = len(nums)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        
        # 初始化：和为0总是可以组成
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if j < nums[i-1]:
                    # 当前元素大于目标和，不能选择
                    dp[i][j] = dp[i-1][j]
                else:
                    # 可以选择或不选择当前元素
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
        
        return dp[n][target]
    
    def canPartition_recursive(self, nums: List[int]) -> bool:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置是否能组成目标和
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n * sum)
        空间复杂度：O(n * sum)
        """
        if not nums:
            return False
        
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        if max(nums) > target:
            return False
        
        memo = {}
        
        def dfs(i, remaining):
            if (i, remaining) in memo:
                return memo[(i, remaining)]
            
            if remaining == 0:
                return True
            
            if i >= len(nums) or remaining < 0:
                return False
            
            # 选择当前元素或不选择
            result = dfs(i + 1, remaining - nums[i]) or dfs(i + 1, remaining)
            memo[(i, remaining)] = result
            return result
        
        return dfs(0, target)
    
    def canPartition_brute_force(self, nums: List[int]) -> bool:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子集
        2. 检查是否存在子集和等于目标和
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return False
        
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        if max(nums) > target:
            return False
        
        def dfs(i, remaining):
            if remaining == 0:
                return True
            
            if i >= len(nums) or remaining < 0:
                return False
            
            # 选择当前元素或不选择
            return dfs(i + 1, remaining - nums[i]) or dfs(i + 1, remaining)
        
        return dfs(0, target)
    
    def canPartition_optimized(self, nums: List[int]) -> bool:
        """
        优化解法：位运算
        
        解题思路：
        1. 使用位运算表示所有可能的和
        2. 通过位运算快速计算
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not nums:
            return False
        
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        if max(nums) > target:
            return False
        
        # 使用位运算表示所有可能的和
        possible_sums = 1  # 初始状态：和为0是可能的
        
        for num in nums:
            # 更新所有可能的和
            possible_sums |= (possible_sums << num)
            
            # 如果目标值已经可达，提前返回
            if possible_sums & (1 << target):
                return True
        
        return False
    
    def canPartition_alternative(self, nums: List[int]) -> bool:
        """
        替代解法：使用集合
        
        解题思路：
        1. 使用集合存储所有可能的和
        2. 逐步更新集合
        
        时间复杂度：O(n * sum)
        空间复杂度：O(sum)
        """
        if not nums:
            return False
        
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        if max(nums) > target:
            return False
        
        # 使用集合存储所有可能的和
        possible_sums = {0}
        
        for num in nums:
            # 创建新的集合，避免重复使用
            new_sums = set()
            for s in possible_sums:
                new_sum = s + num
                if new_sum == target:
                    return True
                if new_sum <= target:
                    new_sums.add(new_sum)
            
            possible_sums.update(new_sums)
        
        return target in possible_sums


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,5,11,5]
    result = solution.canPartition(nums)
    expected = True
    assert result == expected
    
    # 测试用例2
    nums = [1,2,3,5]
    result = solution.canPartition(nums)
    expected = False
    assert result == expected
    
    # 测试用例3
    nums = [1,1,1,1]
    result = solution.canPartition(nums)
    expected = True
    assert result == expected
    
    # 测试用例4
    nums = [1,2,5]
    result = solution.canPartition(nums)
    expected = False
    assert result == expected
    
    # 测试用例5
    nums = [1,1]
    result = solution.canPartition(nums)
    expected = True
    assert result == expected
    
    # 测试二维DP解法
    print("测试二维DP解法...")
    nums = [1,5,11,5]
    result_2d = solution.canPartition_2d(nums)
    assert result_2d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [1,5,11,5]
    result_rec = solution.canPartition_recursive(nums)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,5,11,5]
    result_opt = solution.canPartition_optimized(nums)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    nums = [1,5,11,5]
    result_alt = solution.canPartition_alternative(nums)
    assert result_alt == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
