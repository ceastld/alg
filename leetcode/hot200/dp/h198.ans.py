"""
198. 打家劫舍 - 标准答案
"""
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示偷窃到第i个房屋时的最大金额
        2. 状态转移方程：dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        3. 边界条件：dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
        4. 空间优化：只需要前两个状态
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        # 空间优化：只需要前两个状态
        prev2 = nums[0]  # dp[i-2]
        prev1 = max(nums[0], nums[1])  # dp[i-1]
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def rob_dp_array(self, nums: List[int]) -> int:
        """
        动态规划数组解法
        
        解题思路：
        1. 使用数组存储所有状态
        2. 更直观但空间复杂度为O(n)
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        return dp[-1]
    
    def rob_recursive(self, nums: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        memo = {}
        
        def dfs(i):
            if i in memo:
                return memo[i]
            if i < 0:
                return 0
            if i == 0:
                return nums[0]
            
            memo[i] = max(dfs(i-1), dfs(i-2) + nums[i])
            return memo[i]
        
        return dfs(len(nums) - 1)
    
    def rob_optimized(self, nums: List[int]) -> int:
        """
        优化解法：滚动数组
        
        解题思路：
        1. 使用滚动数组进一步优化空间
        2. 只保存必要的状态信息
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return 0
        
        rob_prev_prev = 0  # 偷窃前前个房屋的最大金额
        rob_prev = 0       # 偷窃前个房屋的最大金额
        
        for num in nums:
            # 当前房屋的最大金额 = max(不偷当前房屋, 偷当前房屋)
            current = max(rob_prev, rob_prev_prev + num)
            rob_prev_prev = rob_prev
            rob_prev = current
        
        return rob_prev
    
    def rob_decision_tree(self, nums: List[int]) -> int:
        """
        决策树解法
        
        解题思路：
        1. 每个房屋有两种选择：偷或不偷
        2. 使用递归枚举所有可能的选择
        3. 返回最大收益
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        def dfs(i, robbed_prev):
            if i >= len(nums):
                return 0
            
            if robbed_prev:
                # 前一个房屋被偷了，当前房屋不能偷
                return dfs(i + 1, False)
            else:
                # 前一个房屋没被偷，当前房屋可以选择偷或不偷
                return max(
                    dfs(i + 1, False),  # 不偷当前房屋
                    dfs(i + 1, True) + nums[i]  # 偷当前房屋
                )
        
        return dfs(0, False)


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,2,3,1]
    result = solution.rob(nums)
    expected = 4
    assert result == expected
    
    # 测试用例2
    nums = [2,7,9,3,1]
    result = solution.rob(nums)
    expected = 12
    assert result == expected
    
    # 测试用例3
    nums = [2,1,1,2]
    result = solution.rob(nums)
    expected = 4
    assert result == expected
    
    # 测试用例4
    nums = [1]
    result = solution.rob(nums)
    expected = 1
    assert result == expected
    
    # 测试用例5
    nums = [1,2]
    result = solution.rob(nums)
    expected = 2
    assert result == expected
    
    # 测试DP数组解法
    print("测试DP数组解法...")
    nums = [2,7,9,3,1]
    result_dp = solution.rob_dp_array(nums)
    expected_dp = 12
    assert result_dp == expected_dp
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [2,7,9,3,1]
    result_rec = solution.rob_recursive(nums)
    expected_rec = 12
    assert result_rec == expected_rec
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [2,7,9,3,1]
    result_opt = solution.rob_optimized(nums)
    expected_opt = 12
    assert result_opt == expected_opt
    
    # 测试决策树解法
    print("测试决策树解法...")
    nums = [2,7,9,3,1]
    result_tree = solution.rob_decision_tree(nums)
    expected_tree = 12
    assert result_tree == expected_tree
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
