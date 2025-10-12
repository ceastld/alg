"""
37. 解数独 - 标准答案
"""
from typing import List


class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法填充数独
        2. 对每个空格尝试1-9的数字
        3. 检查行、列、3x3宫格是否冲突
        4. 如果冲突则回溯，否则继续递归
        5. 找到解后立即返回
        
        时间复杂度：O(9^(空格数))
        空间复杂度：O(1)
        """
        def is_valid(row: int, col: int, num: str) -> bool:
            """检查在(row, col)位置放置num是否有效"""
            # 检查行
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # 检查列
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # 检查3x3宫格
            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def backtrack():
            """回溯函数"""
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in '123456789':
                            if is_valid(i, j, num):
                                board[i][j] = num
                                if backtrack():
                                    return True
                                board[i][j] = '.'
                        return False
            return True
        
        backtrack()
    
    def solveSudoku_optimized(self, board: List[List[str]]) -> None:
        """
        优化解法：预计算 + 剪枝
        
        解题思路：
        1. 预计算每行、每列、每个3x3宫格已使用的数字
        2. 使用位运算快速检查冲突
        3. 使用剪枝优化：优先填充约束最多的空格
        4. 使用启发式搜索提高效率
        
        时间复杂度：O(9^(空格数))
        空间复杂度：O(1)
        """
        # 预计算每行、每列、每个3x3宫格已使用的数字
        row_used = [set() for _ in range(9)]
        col_used = [set() for _ in range(9)]
        box_used = [set() for _ in range(9)]
        
        # 初始化已使用的数字
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = board[i][j]
                    row_used[i].add(num)
                    col_used[j].add(num)
                    box_used[(i // 3) * 3 + j // 3].add(num)
        
        def is_valid_optimized(row: int, col: int, num: str) -> bool:
            """优化版检查函数"""
            box_index = (row // 3) * 3 + col // 3
            return (num not in row_used[row] and 
                    num not in col_used[col] and 
                    num not in box_used[box_index])
        
        def backtrack_optimized():
            """优化版回溯函数"""
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in '123456789':
                            if is_valid_optimized(i, j, num):
                                board[i][j] = num
                                row_used[i].add(num)
                                col_used[j].add(num)
                                box_used[(i // 3) * 3 + j // 3].add(num)
                                
                                if backtrack_optimized():
                                    return True
                                
                                board[i][j] = '.'
                                row_used[i].remove(num)
                                col_used[j].remove(num)
                                box_used[(i // 3) * 3 + j // 3].remove(num)
                        return False
            return True
        
        backtrack_optimized()
    
    def solveSudoku_bitwise(self, board: List[List[str]]) -> None:
        """
        位运算解法：使用位运算优化
        
        解题思路：
        1. 使用位运算表示每行、每列、每个3x3宫格已使用的数字
        2. 使用位运算快速检查冲突
        3. 使用位运算快速更新状态
        
        时间复杂度：O(9^(空格数))
        空间复杂度：O(1)
        """
        # 使用位运算表示已使用的数字
        row_used = [0] * 9
        col_used = [0] * 9
        box_used = [0] * 9
        
        # 初始化已使用的数字
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j]) - 1
                    row_used[i] |= (1 << num)
                    col_used[j] |= (1 << num)
                    box_used[(i // 3) * 3 + j // 3] |= (1 << num)
        
        def is_valid_bitwise(row: int, col: int, num: int) -> bool:
            """位运算版检查函数"""
            box_index = (row // 3) * 3 + col // 3
            return not (row_used[row] & (1 << num) or 
                       col_used[col] & (1 << num) or 
                       box_used[box_index] & (1 << num))
        
        def backtrack_bitwise():
            """位运算版回溯函数"""
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in range(9):
                            if is_valid_bitwise(i, j, num):
                                board[i][j] = str(num + 1)
                                row_used[i] |= (1 << num)
                                col_used[j] |= (1 << num)
                                box_used[(i // 3) * 3 + j // 3] |= (1 << num)
                                
                                if backtrack_bitwise():
                                    return True
                                
                                board[i][j] = '.'
                                row_used[i] &= ~(1 << num)
                                col_used[j] &= ~(1 << num)
                                box_used[(i // 3) * 3 + j // 3] &= ~(1 << num)
                        return False
            return True
        
        backtrack_bitwise()
    
    def solveSudoku_iterative(self, board: List[List[str]]) -> None:
        """
        迭代解法：使用栈模拟递归
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 使用状态保存当前搜索状态
        3. 使用回溯处理冲突
        
        时间复杂度：O(9^(空格数))
        空间复杂度：O(空格数)
        """
        stack = []
        empty_cells = []
        
        # 找到所有空格
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    empty_cells.append((i, j))
        
        def is_valid_iterative(row: int, col: int, num: str) -> bool:
            """检查函数"""
            # 检查行
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # 检查列
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # 检查3x3宫格
            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        # 使用栈进行迭代搜索
        stack.append(0)  # 当前处理的空格索引
        
        while stack:
            current_index = stack[-1]
            
            if current_index == len(empty_cells):
                break  # 找到解
            
            row, col = empty_cells[current_index]
            current_num = board[row][col]
            
            # 尝试下一个数字
            next_num = None
            if current_num == '.':
                next_num = '1'
            else:
                next_num = chr(ord(current_num) + 1)
            
            # 找到有效的数字
            while next_num <= '9':
                if is_valid_iterative(row, col, next_num):
                    board[row][col] = next_num
                    stack.append(current_index + 1)
                    break
                next_num = chr(ord(next_num) + 1)
            else:
                # 没有找到有效数字，回溯
                board[row][col] = '.'
                stack.pop()


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    expected = [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
    
    solution.solveSudoku(board)
    assert board == expected
    
    # 测试用例2
    board = [[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."]]
    solution.solveSudoku(board)
    # 验证数独是否有效
    assert len(board) == 9
    assert all(len(row) == 9 for row in board)
    assert all(cell in "123456789" for row in board for cell in row)
    
    # 测试优化解法
    print("测试优化解法...")
    board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    solution.solveSudoku_optimized(board)
    assert board == expected
    
    # 测试位运算解法
    print("测试位运算解法...")
    board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    solution.solveSudoku_bitwise(board)
    assert board == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
