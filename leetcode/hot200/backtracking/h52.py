"""
52. N皇后II
n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。

题目链接：https://leetcode.cn/problems/n-queens-ii/

示例 1:
输入：n = 4
输出：2
解释：如上图所示，4 皇后问题存在两个不同的解法。

示例 2:
输入：n = 1
输出：1

提示：
- 1 <= n <= 9
"""

from typing import List


class Solution:
    def totalNQueens(self, n: int) -> int:
        """
        只计数N皇后问题的解的数量
        
        解题思路：
        1. 使用位运算优化对角线检查
        2. 使用三个整数表示列、主对角线、副对角线的占用情况
        3. 通过位运算快速检查冲突
        4. 递归回溯，只计数不存储具体解
        
        时间复杂度：O(n!)
        空间复杂度：O(n)
        """
        def dfs(row: int, cols: int, diag1: int, diag2: int) -> int:
            if row == n:
                return 1
            
            count = 0
            # 计算可用的列
            available = ((1 << n) - 1) & (~(cols | diag1 | diag2))
            
            while available:
                # 获取最低位的1
                pos = available & (-available)
                # 清除最低位的1
                available &= (available - 1)
                
                # 递归计算
                count += dfs(row + 1, 
                           cols | pos, 
                           (diag1 | pos) << 1, 
                           (diag2 | pos) >> 1)
            
            return count
        
        return dfs(0, 0, 0, 0)
    
    def totalNQueens1(self, n: int) -> int:
        arr = [1, 0, 0, 2, 10, 4, 40, 92, 352]
        return arr[n-1]

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    n = 4
    result = solution.totalNQueens(n)
    expected = 2
    assert result == expected
    
    # 测试用例2
    n = 1
    result = solution.totalNQueens(n)
    expected = 1
    assert result == expected
    
    # 测试用例3
    n = 2
    result = solution.totalNQueens(n)
    expected = 0
    assert result == expected
    
    # 测试用例4
    n = 3
    result = solution.totalNQueens(n)
    expected = 0
    assert result == expected
    
    # 测试用例5
    n = 5
    result = solution.totalNQueens(n)
    expected = 10
    assert result == expected
    
    # 测试用例6
    n = 6
    result = solution.totalNQueens(n)
    expected = 4
    assert result == expected
    
    # 测试用例7
    n = 7
    result = solution.totalNQueens(n)
    expected = 40
    assert result == expected
    
    # 测试用例8
    n = 8
    result = solution.totalNQueens(n)
    expected = 92
    assert result == expected
    
    # 测试用例9
    n = 9
    result = solution.totalNQueens(n)
    expected = 352
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
