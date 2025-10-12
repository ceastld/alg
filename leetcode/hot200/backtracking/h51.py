"""
51. N皇后
按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

题目链接：https://leetcode.cn/problems/n-queens/

示例 1:
输入: n = 4
输出: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释: 如上图所示，4 皇后问题存在两个不同的解法。

示例 2:
输入: n = 1
输出: [["Q"]]

提示：
- 1 <= n <= 9
"""

from typing import List


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:

        def convert(path: list[int]) -> list[str]:
            return ["." * i + "Q" + "." * (n - i - 1) for i in path]

        def dfs(row: int, path: list[int]):
            if row == n:
                result.append(convert(path))
                return
            for col in range(n):
                if col in path:
                    continue
                flag = all(abs(r - row) != abs(c - col) for r, c in enumerate(path))
                if flag:
                    path.append(col)
                    dfs(row + 1, path)
                    path.pop()

        result = []
        path = []
        dfs(0, path)
        return result

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


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    n = 4
    result = solution.solveNQueens(n)
    expected = [[".Q..", "...Q", "Q...", "..Q."], ["..Q.", "Q...", "...Q", ".Q.."]]
    assert len(result) == len(expected)

    # 测试用例2
    n = 1
    result = solution.solveNQueens(n)
    expected = [["Q"]]
    assert result == expected

    # 测试用例3
    n = 2
    result = solution.solveNQueens(n)
    expected = []
    assert result == expected

    print("所有测试用例通过！")
    
    # 输出1~20的N皇后问题解的数量
    print("\nN皇后问题解的数量 (1~20):")
    print("n\t解的数量")
    print("-" * 20)
    for i in range(1, 10):
        count = solution.totalNQueens(i)
        print(f"{i}\t{count}")


if __name__ == "__main__":
    main()
