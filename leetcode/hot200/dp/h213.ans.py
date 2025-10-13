"""
213. 打家劫舍II - 标准答案
"""
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        标准解法：分情况讨论
        
        解题思路：
        1. 由于房屋围成环形，第一个和最后一个相邻
        2. 分两种情况：
           - 不偷第一个房屋：考虑nums[1:]
           - 不偷最后一个房屋：考虑nums[:-1]
        3. 取两种情况的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        def rob_linear(houses):
            """线性房屋的打家劫舍"""
            if not houses:
                return 0
            if len(houses) == 1:
                return houses[0]
            
            prev2 = houses[0]
            prev1 = max(houses[0], houses[1])
            
            for i in range(2, len(houses)):
                current = max(prev1, prev2 + houses[i])
                prev2 = prev1
                prev1 = current
            
            return prev1
        
        # 情况1：不偷第一个房屋
        case1 = rob_linear(nums[1:])
        # 情况2：不偷最后一个房屋
        case2 = rob_linear(nums[:-1])
        
        return max(case1, case2)
    
    def rob_dp(self, nums: List[int]) -> int:
        """
        动态规划解法
        
        解题思路：
        1. 使用两个DP数组分别处理两种情况
        2. dp1[i] 表示不偷第一个房屋时，前i个房屋的最大金额
        3. dp2[i] 表示不偷最后一个房屋时，前i个房屋的最大金额
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        n = len(nums)
        
        # 情况1：不偷第一个房屋
        dp1 = [0] * (n - 1)
        dp1[0] = nums[1]
        if n > 2:
            dp1[1] = max(nums[1], nums[2])
            for i in range(2, n - 1):
                dp1[i] = max(dp1[i-1], dp1[i-2] + nums[i+1])
        
        # 情况2：不偷最后一个房屋
        dp2 = [0] * (n - 1)
        dp2[0] = nums[0]
        if n > 2:
            dp2[1] = max(nums[0], nums[1])
            for i in range(2, n - 1):
                dp2[i] = max(dp2[i-1], dp2[i-2] + nums[i])
        
        return max(dp1[-1], dp2[-1])
    
    def rob_recursive(self, nums: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算两种情况的最大值
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        memo = {}
        
        def dfs(start, end):
            if (start, end) in memo:
                return memo[(start, end)]
            
            if start > end:
                return 0
            if start == end:
                return nums[start]
            
            # 选择偷当前房屋
            rob_current = nums[start] + dfs(start + 2, end)
            # 选择不偷当前房屋
            skip_current = dfs(start + 1, end)
            
            result = max(rob_current, skip_current)
            memo[(start, end)] = result
            return result
        
        # 情况1：不偷第一个房屋
        case1 = dfs(1, len(nums) - 1)
        # 情况2：不偷最后一个房屋
        case2 = dfs(0, len(nums) - 2)
        
        return max(case1, case2)
    
    def rob_optimized(self, nums: List[int]) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用滚动数组优化空间复杂度
        2. 分别处理两种情况
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        def rob_range(start, end):
            """计算指定范围内的最大金额"""
            if start > end:
                return 0
            
            prev2 = 0
            prev1 = nums[start]
            
            for i in range(start + 1, end + 1):
                current = max(prev1, prev2 + nums[i])
                prev2 = prev1
                prev1 = current
            
            return prev1
        
        # 情况1：不偷第一个房屋
        case1 = rob_range(1, len(nums) - 1)
        # 情况2：不偷最后一个房屋
        case2 = rob_range(0, len(nums) - 2)
        
        return max(case1, case2)
    
    def rob_brute_force(self, nums: List[int]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的偷窃方案
        2. 检查每个方案是否合法
        3. 返回最大金额
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        max_amount = 0
        n = len(nums)
        
        # 枚举所有可能的方案
        for mask in range(1 << n):
            amount = 0
            valid = True
            
            for i in range(n):
                if mask & (1 << i):  # 如果选择偷第i个房屋
                    amount += nums[i]
                    # 检查是否与相邻房屋冲突
                    if (mask & (1 << ((i + 1) % n))) or (mask & (1 << ((i - 1) % n))):
                        valid = False
                        break
            
            if valid:
                max_amount = max(max_amount, amount)
        
        return max_amount


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [2,3,2]
    result = solution.rob(nums)
    expected = 3
    assert result == expected
    
    # 测试用例2
    nums = [1,2,3,1]
    result = solution.rob(nums)
    expected = 4
    assert result == expected
    
    # 测试用例3
    nums = [1,2,3]
    result = solution.rob(nums)
    expected = 3
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
    
    # 测试DP解法
    print("测试DP解法...")
    nums = [2,3,2]
    result_dp = solution.rob_dp(nums)
    assert result_dp == expected
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [2,3,2]
    result_rec = solution.rob_recursive(nums)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [2,3,2]
    result_opt = solution.rob_optimized(nums)
    assert result_opt == expected
    
    # 测试暴力解法
    print("测试暴力解法...")
    nums = [2,3,2]
    result_bf = solution.rob_brute_force(nums)
    assert result_bf == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
