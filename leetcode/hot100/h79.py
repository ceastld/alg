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
