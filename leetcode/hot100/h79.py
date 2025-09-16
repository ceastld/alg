"""
LeetCode 79. Word Search

题目描述：
给定一个m x n二维字符网格board和一个字符串单词word。如果word存在于网格中，返回true；否则，返回false。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中"相邻"单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

示例：
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

数据范围：
- m == board.length
- n = board[i].length
- 1 <= m, n <= 6
- 1 <= word.length <= 15
- board和word仅由大小写英文字母组成
"""

class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        def dfs(row, col, index):
            if index == len(word):
                return True
            
            if (row < 0 or row >= len(board) or col < 0 or col >= len(board[0]) or 
                board[row][col] != word[index]):
                return False
            
            # 标记已访问
            temp = board[row][col]
            board[row][col] = '#'
            
            # 四个方向搜索
            found = (dfs(row + 1, col, index + 1) or
                    dfs(row - 1, col, index + 1) or
                    dfs(row, col + 1, index + 1) or
                    dfs(row, col - 1, index + 1))
            
            # 恢复原值
            board[row][col] = temp
            
            return found
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        
        return False
