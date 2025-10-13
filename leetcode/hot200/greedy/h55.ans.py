"""
55. 跳跃游戏 - 标准答案
"""
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 维护一个变量max_reach，表示当前能到达的最远位置
        2. 遍历数组，更新max_reach = max(max_reach, i + nums[i])
        3. 如果当前位置i > max_reach，说明无法到达当前位置，返回False
        4. 如果max_reach >= len(nums)-1，说明能到达最后一个位置
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return False
        
        max_reach = 0
        
        for i in range(len(nums)):
            # 如果当前位置超出了能到达的最远位置
            if i > max_reach:
                return False
            
            # 更新能到达的最远位置
            max_reach = max(max_reach, i + nums[i])
            
            # 如果能到达最后一个位置，提前返回
            if max_reach >= len(nums) - 1:
                return True
        
        return max_reach >= len(nums) - 1
    
    def canJump_alternative(self, nums: List[int]) -> bool:
        """
        替代解法：从后往前遍历
        
        解题思路：
        1. 从最后一个位置开始往前遍历
        2. 维护一个变量last_pos，表示能到达最后一个位置的最远位置
        3. 如果当前位置i + nums[i] >= last_pos，则更新last_pos = i
        4. 最后检查last_pos是否为0
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return False
        
        last_pos = len(nums) - 1
        
        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= last_pos:
                last_pos = i
        
        return last_pos == 0
    
    def canJump_dp(self, nums: List[int]) -> bool:
        """
        动态规划解法：记忆化搜索
        
        解题思路：
        1. dp[i] 表示从位置i是否能到达最后一个位置
        2. 状态转移：dp[i] = True if i + nums[i] >= len(nums)-1 or any(dp[j] for j in range(i+1, i+nums[i]+1))
        3. 从后往前填充dp数组
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not nums:
            return False
        
        n = len(nums)
        dp = [False] * n
        dp[n-1] = True
        
        for i in range(n-2, -1, -1):
            # 检查从位置i能到达的所有位置
            for j in range(i+1, min(i+nums[i]+1, n)):
                if dp[j]:
                    dp[i] = True
                    break
        
        return dp[0]
    
    def canJump_optimized(self, nums: List[int]) -> bool:
        """
        优化解法：贪心算法（简化版）
        
        解题思路：
        1. 只维护一个变量max_reach
        2. 遍历时检查当前位置是否可达
        3. 更新max_reach并检查是否到达终点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        max_reach = 0
        
        for i, jump in enumerate(nums):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + jump)
        
        return True


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.canJump([2,3,1,1,4]) == True
    
    # 测试用例2
    assert solution.canJump([3,2,1,0,4]) == False
    
    # 测试用例3
    assert solution.canJump([0]) == True
    
    # 测试用例4
    assert solution.canJump([1,0,1,0]) == False
    
    # 测试用例5
    assert solution.canJump([2,0,0]) == True
    
    # 测试用例6：边界情况
    assert solution.canJump([]) == False
    assert solution.canJump([1]) == True
    
    # 测试用例7：大跳跃
    assert solution.canJump([5,4,3,2,1,0,0]) == False
    assert solution.canJump([5,4,3,2,1,0,1]) == True
    
    # 测试用例8：全零数组
    assert solution.canJump([0,0,0,0]) == False
    
    # 测试用例9：单步跳跃
    assert solution.canJump([1,1,1,1]) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
