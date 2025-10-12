"""
51. N皇后 - 标准答案
"""
from typing import List


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法逐行放置皇后
        2. 使用三个集合记录已占用的列、主对角线、副对角线
        3. 对于每一行，尝试在每一列放置皇后
        4. 检查是否与已放置的皇后冲突
        5. 如果冲突，跳过；如果不冲突，继续递归
        6. 当所有行都放置了皇后时，将结果加入答案
        
        时间复杂度：O(n!)
        空间复杂度：O(n)
        """
        result = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        
        def is_safe(row, col):
            # 检查列
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            # 检查主对角线（左上到右下）
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            
            # 检查副对角线（右上到左下）
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            
            return True
        
        def backtrack(row):
            # 如果所有行都放置了皇后，加入结果
            if row == n:
                result.append([''.join(row) for row in board])
                return
            
            # 尝试在当前行的每一列放置皇后
            for col in range(n):
                if is_safe(row, col):
                    board[row][col] = 'Q'
                    backtrack(row + 1)
                    board[row][col] = '.'
        
        backtrack(0)
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    n = 4
    result = solution.solveNQueens(n)
    expected = [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
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


if __name__ == "__main__":
    main()
