"""
45. 跳跃游戏II - 标准答案
"""
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 维护三个变量：
           - jumps: 跳跃次数
           - current_end: 当前跳跃能到达的最远位置
           - farthest: 所有位置中能到达的最远位置
        2. 遍历数组，更新farthest
        3. 当到达current_end时，必须进行一次跳跃，更新jumps和current_end
        4. 贪心策略：每次跳跃都选择能到达最远位置的位置
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) <= 1:
            return 0
        
        jumps = 0
        current_end = 0  # 当前跳跃能到达的最远位置
        farthest = 0     # 所有位置中能到达的最远位置
        
        for i in range(len(nums) - 1):  # 不需要检查最后一个位置
            # 更新能到达的最远位置
            farthest = max(farthest, i + nums[i])
            
            # 如果到达了当前跳跃的边界，必须进行下一次跳跃
            if i == current_end:
                jumps += 1
                current_end = farthest
                
                # 如果能到达最后一个位置，提前返回
                if current_end >= len(nums) - 1:
                    break
        
        return jumps
    
    def jump_alternative(self, nums: List[int]) -> int:
        """
        替代解法：贪心算法（从后往前）
        
        解题思路：
        1. 从最后一个位置开始往前找
        2. 找到能直接到达当前位置的最远位置
        3. 更新当前位置为找到的位置，跳跃次数+1
        4. 重复直到到达起始位置
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if len(nums) <= 1:
            return 0
        
        jumps = 0
        position = len(nums) - 1
        
        while position > 0:
            # 找到能直接到达当前位置的最远位置
            for i in range(position):
                if i + nums[i] >= position:
                    position = i
                    jumps += 1
                    break
        
        return jumps
    
    def jump_dp(self, nums: List[int]) -> int:
        """
        动态规划解法：记忆化搜索
        
        解题思路：
        1. dp[i] 表示从位置i到最后一个位置的最少跳跃次数
        2. 状态转移：dp[i] = min(dp[j] + 1) for j in range(i+1, i+nums[i]+1)
        3. 从后往前填充dp数组
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if len(nums) <= 1:
            return 0
        
        n = len(nums)
        dp = [float('inf')] * n
        dp[n-1] = 0
        
        for i in range(n-2, -1, -1):
            # 检查从位置i能到达的所有位置
            for j in range(i+1, min(i+nums[i]+1, n)):
                dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[0]
    
    def jump_bfs(self, nums: List[int]) -> int:
        """
        BFS解法：层序遍历
        
        解题思路：
        1. 将跳跃过程看作BFS的层序遍历
        2. 每一层代表一次跳跃能到达的所有位置
        3. 使用队列维护当前层的所有位置
        4. 当队列为空或到达最后一个位置时结束
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if len(nums) <= 1:
            return 0
        
        from collections import deque
        
        queue = deque([0])
        jumps = 0
        visited = {0}
        
        while queue:
            size = len(queue)
            
            for _ in range(size):
                current = queue.popleft()
                
                # 检查从当前位置能到达的所有位置
                for i in range(current + 1, min(current + nums[current] + 1, len(nums))):
                    if i == len(nums) - 1:
                        return jumps + 1
                    
                    if i not in visited:
                        visited.add(i)
                        queue.append(i)
            
            jumps += 1
        
        return jumps
    
    def jump_optimized(self, nums: List[int]) -> int:
        """
        优化解法：贪心算法（简化版）
        
        解题思路：
        1. 使用两个指针：left和right
        2. left表示当前层的起始位置，right表示当前层的结束位置
        3. 遍历当前层，找到下一层能到达的最远位置
        4. 更新left和right，跳跃次数+1
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) <= 1:
            return 0
        
        jumps = 0
        left = 0
        right = 0
        
        while right < len(nums) - 1:
            farthest = 0
            # 在当前层中找到能到达的最远位置
            for i in range(left, right + 1):
                farthest = max(farthest, i + nums[i])
            
            left = right + 1
            right = farthest
            jumps += 1
        
        return jumps


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.jump([2,3,1,1,4]) == 2
    
    # 测试用例2
    assert solution.jump([2,3,0,1,4]) == 2
    
    # 测试用例3
    assert solution.jump([1,2,3]) == 2
    
    # 测试用例4
    assert solution.jump([1]) == 0
    
    # 测试用例5
    assert solution.jump([2,1]) == 1
    
    # 测试用例6：边界情况
    assert solution.jump([1,1,1,1]) == 3
    assert solution.jump([2,1,1,1,1]) == 2
    
    # 测试用例7：大跳跃
    assert solution.jump([5,4,3,2,1,0,1]) == 2
    assert solution.jump([1,2,1,1,1]) == 3
    
    # 测试用例8：单步跳跃
    assert solution.jump([1,1,1,1,1]) == 4
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
