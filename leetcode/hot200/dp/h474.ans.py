"""
474. 一和零 - 标准答案
"""
from typing import List


class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """
        标准解法：动态规划（二维0-1背包）
        
        解题思路：
        1. 问题转化为：在限制0和1的数量的情况下，选择最多的字符串
        2. 使用二维DP：dp[i][j] 表示最多有i个0和j个1时能选择的最大字符串数
        3. 状态转移方程：
           - dp[i][j] = max(dp[i][j], dp[i-zeros][j-ones] + 1)
        
        时间复杂度：O(len(strs) * m * n)
        空间复杂度：O(m * n)
        """
        if not strs:
            return 0
        
        # 使用二维数组优化空间
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for s in strs:
            # 统计当前字符串中0和1的个数
            zeros = s.count('0')
            ones = s.count('1')
            
            # 从后往前遍历，避免重复使用
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        
        return dp[m][n]
    
    def findMaxForm_3d(self, strs: List[str], m: int, n: int) -> int:
        """
        三维DP解法
        
        解题思路：
        1. dp[k][i][j] 表示前k个字符串中最多有i个0和j个1时能选择的最大字符串数
        2. 状态转移方程：
           - dp[k][i][j] = max(dp[k-1][i][j], dp[k-1][i-zeros][j-ones] + 1)
        
        时间复杂度：O(len(strs) * m * n)
        空间复杂度：O(len(strs) * m * n)
        """
        if not strs:
            return 0
        
        length = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(length + 1)]
        
        for k in range(1, length + 1):
            s = strs[k - 1]
            zeros = s.count('0')
            ones = s.count('1')
            
            for i in range(m + 1):
                for j in range(n + 1):
                    if i >= zeros and j >= ones:
                        dp[k][i][j] = max(dp[k-1][i][j], dp[k-1][i-zeros][j-ones] + 1)
                    else:
                        dp[k][i][j] = dp[k-1][i][j]
        
        return dp[length][m][n]
    
    def findMaxForm_recursive(self, strs: List[str], m: int, n: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最大字符串数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(len(strs) * m * n)
        空间复杂度：O(len(strs) * m * n)
        """
        if not strs:
            return 0
        
        memo = {}
        
        def dfs(i, remaining_zeros, remaining_ones):
            if (i, remaining_zeros, remaining_ones) in memo:
                return memo[(i, remaining_zeros, remaining_ones)]
            
            if i >= len(strs):
                return 0
            
            s = strs[i]
            zeros = s.count('0')
            ones = s.count('1')
            
            # 选择当前字符串或不选择
            if zeros <= remaining_zeros and ones <= remaining_ones:
                result = max(dfs(i + 1, remaining_zeros - zeros, remaining_ones - ones) + 1,
                           dfs(i + 1, remaining_zeros, remaining_ones))
            else:
                result = dfs(i + 1, remaining_zeros, remaining_ones)
            
            memo[(i, remaining_zeros, remaining_ones)] = result
            return result
        
        return dfs(0, m, n)
    
    def findMaxForm_brute_force(self, strs: List[str], m: int, n: int) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子集
        2. 计算每种子集的0和1的个数
        3. 返回满足条件的最大子集大小
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not strs:
            return 0
        
        max_count = 0
        
        def dfs(i, selected, remaining_zeros, remaining_ones):
            nonlocal max_count
            
            if i == len(strs):
                max_count = max(max_count, len(selected))
                return
            
            s = strs[i]
            zeros = s.count('0')
            ones = s.count('1')
            
            # 选择当前字符串
            if zeros <= remaining_zeros and ones <= remaining_ones:
                dfs(i + 1, selected + [s], remaining_zeros - zeros, remaining_ones - ones)
            
            # 不选择当前字符串
            dfs(i + 1, selected, remaining_zeros, remaining_ones)
        
        dfs(0, [], m, n)
        return max_count
    
    def findMaxForm_optimized(self, strs: List[str], m: int, n: int) -> int:
        """
        优化解法：预处理统计
        
        解题思路：
        1. 预处理每个字符串中0和1的个数
        2. 使用二维DP优化空间
        
        时间复杂度：O(len(strs) * m * n)
        空间复杂度：O(m * n)
        """
        if not strs:
            return 0
        
        # 预处理统计每个字符串中0和1的个数
        counts = []
        for s in strs:
            zeros = s.count('0')
            ones = s.count('1')
            counts.append((zeros, ones))
        
        # 使用二维数组优化空间
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for zeros, ones in counts:
            # 从后往前遍历，避免重复使用
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        
        return dp[m][n]
    
    def findMaxForm_alternative(self, strs: List[str], m: int, n: int) -> int:
        """
        替代解法：使用字典
        
        解题思路：
        1. 使用字典存储所有可能的状态
        2. 逐步更新状态
        
        时间复杂度：O(len(strs) * m * n)
        空间复杂度：O(m * n)
        """
        if not strs:
            return 0
        
        # 使用字典存储所有可能的状态
        dp = {(0, 0): 0}  # 初始状态：0个0和0个1时能选择0个字符串
        
        for s in strs:
            zeros = s.count('0')
            ones = s.count('1')
            
            # 创建新的状态字典
            new_dp = {}
            
            for (i, j), count in dp.items():
                # 不选择当前字符串
                new_dp[(i, j)] = max(new_dp.get((i, j), 0), count)
                
                # 选择当前字符串
                new_i = i + zeros
                new_j = j + ones
                if new_i <= m and new_j <= n:
                    new_dp[(new_i, new_j)] = max(new_dp.get((new_i, new_j), 0), count + 1)
            
            dp = new_dp
        
        return max(dp.values()) if dp else 0
    
    def findMaxForm_greedy(self, strs: List[str], m: int, n: int) -> int:
        """
        贪心解法：优先选择短字符串
        
        解题思路：
        1. 优先选择0和1个数较少的字符串
        2. 贪心地选择最优解
        
        时间复杂度：O(len(strs) * log(len(strs)))
        空间复杂度：O(len(strs))
        """
        if not strs:
            return 0
        
        # 按0和1的总数排序
        sorted_strs = sorted(strs, key=lambda s: s.count('0') + s.count('1'))
        
        selected = []
        remaining_zeros = m
        remaining_ones = n
        
        for s in sorted_strs:
            zeros = s.count('0')
            ones = s.count('1')
            
            if zeros <= remaining_zeros and ones <= remaining_ones:
                selected.append(s)
                remaining_zeros -= zeros
                remaining_ones -= ones
        
        return len(selected)
    
    def findMaxForm_dfs(self, strs: List[str], m: int, n: int) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的子集
        2. 统计满足条件的最大子集大小
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not strs:
            return 0
        
        max_count = 0
        
        def dfs(i, selected, remaining_zeros, remaining_ones):
            nonlocal max_count
            
            if i == len(strs):
                max_count = max(max_count, len(selected))
                return
            
            s = strs[i]
            zeros = s.count('0')
            ones = s.count('1')
            
            # 选择当前字符串
            if zeros <= remaining_zeros and ones <= remaining_ones:
                dfs(i + 1, selected + [s], remaining_zeros - zeros, remaining_ones - ones)
            
            # 不选择当前字符串
            dfs(i + 1, selected, remaining_zeros, remaining_ones)
        
        dfs(0, [], m, n)
        return max_count


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    strs = ["10","0001","111001","1","0"]
    m = 5
    n = 3
    result = solution.findMaxForm(strs, m, n)
    expected = 4
    assert result == expected
    
    # 测试用例2
    strs = ["10","0","1"]
    m = 1
    n = 1
    result = solution.findMaxForm(strs, m, n)
    expected = 2
    assert result == expected
    
    # 测试用例3
    strs = ["10","0001","111001","1","0"]
    m = 3
    n = 2
    result = solution.findMaxForm(strs, m, n)
    expected = 3
    assert result == expected
    
    # 测试用例4
    strs = ["10","0001","111001","1","0"]
    m = 1
    n = 1
    result = solution.findMaxForm(strs, m, n)
    expected = 2
    assert result == expected
    
    # 测试用例5
    strs = ["10","0001","111001","1","0"]
    m = 0
    n = 0
    result = solution.findMaxForm(strs, m, n)
    expected = 0
    assert result == expected
    
    # 测试三维DP解法
    print("测试三维DP解法...")
    strs = ["10","0001","111001","1","0"]
    m = 5
    n = 3
    result_3d = solution.findMaxForm_3d(strs, m, n)
    assert result_3d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    strs = ["10","0001","111001","1","0"]
    m = 5
    n = 3
    result_rec = solution.findMaxForm_recursive(strs, m, n)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    strs = ["10","0001","111001","1","0"]
    m = 5
    n = 3
    result_opt = solution.findMaxForm_optimized(strs, m, n)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    strs = ["10","0001","111001","1","0"]
    m = 5
    n = 3
    result_alt = solution.findMaxForm_alternative(strs, m, n)
    assert result_alt == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    strs = ["10","0001","111001","1","0"]
    m = 5
    n = 3
    result_greedy = solution.findMaxForm_greedy(strs, m, n)
    assert result_greedy == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
