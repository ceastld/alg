"""
135. 分发糖果 - 标准答案
"""
from typing import List


class Solution:
    def candy(self, ratings: List[int]) -> int:
        """
        标准解法：贪心算法（两次遍历）
        
        解题思路：
        1. 首先从左到右遍历，确保每个孩子比左边评分低的孩子获得更多糖果
        2. 然后从右到左遍历，确保每个孩子比右边评分低的孩子获得更多糖果
        3. 贪心策略：每次只考虑一个方向的关系，最后取最大值
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(ratings)
        if n <= 1:
            return n
        
        # 初始化每个孩子至少1个糖果
        candies = [1] * n
        
        # 从左到右遍历，确保每个孩子比左边评分低的孩子获得更多糖果
        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1
        
        # 从右到左遍历，确保每个孩子比右边评分低的孩子获得更多糖果
        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i], candies[i+1] + 1)
        
        return sum(candies)
    
    def candy_alternative(self, ratings: List[int]) -> int:
        """
        替代解法：贪心算法（一次遍历）
        
        解题思路：
        1. 使用一个变量记录当前递减序列的长度
        2. 当遇到递增时，正常分配糖果
        3. 当遇到递减时，需要回溯调整之前的分配
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(ratings)
        if n <= 1:
            return n
        
        total = 1  # 第一个孩子至少1个糖果
        up = 0     # 当前递增序列的长度
        down = 0   # 当前递减序列的长度
        peak = 0    # 峰值位置
        
        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                # 递增序列
                up += 1
                down = 0
                peak = up
                total += up + 1
            elif ratings[i] < ratings[i-1]:
                # 递减序列
                down += 1
                up = 0
                total += down + 1
                # 如果峰值不够高，需要调整
                if peak >= down:
                    total -= 1
            else:
                # 相等，重置
                up = 0
                down = 0
                peak = 0
                total += 1
        
        return total
    
    def candy_brute_force(self, ratings: List[int]) -> int:
        """
        暴力解法：动态规划
        
        解题思路：
        1. dp[i] 表示第i个孩子应该获得的糖果数
        2. 状态转移：dp[i] = max(dp[i-1] + 1, dp[i+1] + 1) if ratings[i] > ratings[i-1] or ratings[i] > ratings[i+1]
        3. 需要多次迭代直到收敛
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        n = len(ratings)
        if n <= 1:
            return n
        
        candies = [1] * n
        changed = True
        
        while changed:
            changed = False
            for i in range(n):
                # 检查与左边的关系
                if i > 0 and ratings[i] > ratings[i-1] and candies[i] <= candies[i-1]:
                    candies[i] = candies[i-1] + 1
                    changed = True
                
                # 检查与右边的关系
                if i < n-1 and ratings[i] > ratings[i+1] and candies[i] <= candies[i+1]:
                    candies[i] = candies[i+1] + 1
                    changed = True
        
        return sum(candies)
    
    def candy_optimized(self, ratings: List[int]) -> int:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 使用两个变量记录当前递增和递减序列的长度
        2. 根据序列长度计算需要的糖果数
        3. 处理边界情况
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(ratings)
        if n <= 1:
            return n
        
        total = 1
        up = 0
        down = 0
        peak = 0
        
        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                # 递增序列
                up += 1
                down = 0
                peak = up
                total += up + 1
            elif ratings[i] < ratings[i-1]:
                # 递减序列
                down += 1
                up = 0
                total += down + 1
                # 如果峰值不够高，需要调整
                if peak >= down:
                    total -= 1
            else:
                # 相等，重置
                up = 0
                down = 0
                peak = 0
                total += 1
        
        return total
    
    def candy_detailed(self, ratings: List[int]) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 从左到右遍历，处理递增关系
        2. 从右到左遍历，处理递减关系
        3. 取两次遍历的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(ratings)
        if n <= 1:
            return n
        
        # 初始化每个孩子至少1个糖果
        left_to_right = [1] * n
        right_to_left = [1] * n
        
        # 从左到右遍历，处理递增关系
        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                left_to_right[i] = left_to_right[i-1] + 1
        
        # 从右到左遍历，处理递减关系
        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                right_to_left[i] = right_to_left[i+1] + 1
        
        # 取两次遍历的最大值
        total = 0
        for i in range(n):
            total += max(left_to_right[i], right_to_left[i])
        
        return total


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.candy([1,0,2]) == 5
    
    # 测试用例2
    assert solution.candy([1,2,2]) == 4
    
    # 测试用例3
    assert solution.candy([1,2,3,4,5]) == 15
    
    # 测试用例4
    assert solution.candy([5,4,3,2,1]) == 15
    
    # 测试用例5
    assert solution.candy([1,3,2,2,1]) == 7
    
    # 测试用例6：边界情况
    assert solution.candy([1]) == 1
    assert solution.candy([1,1]) == 2
    assert solution.candy([1,2]) == 3
    
    # 测试用例7：相等评分
    assert solution.candy([1,1,1]) == 3
    assert solution.candy([2,2,2,2]) == 4
    
    # 测试用例8：复杂情况
    assert solution.candy([1,2,3,2,1]) == 9
    assert solution.candy([1,3,4,5,2]) == 11
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
