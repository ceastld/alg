"""
52. N皇后II - 标准答案
"""
from typing import List


class Solution:
    def totalNQueens(self, n: int) -> int:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的N皇后解
        2. 使用数组记录每行、每列、每个对角线的占用情况
        3. 递归处理每一行，尝试放置皇后
        4. 使用剪枝优化：如果当前行无法放置皇后，则提前终止
        5. 只计数不存储具体解，提高效率
        
        时间复杂度：O(n!)
        空间复杂度：O(n)
        """
        result = 0
        
        def backtrack(row: int, cols: List[int], diag1: List[int], diag2: List[int]):
            nonlocal result
            
            if row == n:
                result += 1
                return
            
            for col in range(n):
                # 检查列是否被占用
                if cols[col]:
                    continue
                
                # 检查主对角线是否被占用
                if diag1[row - col + n - 1]:
                    continue
                
                # 检查副对角线是否被占用
                if diag2[row + col]:
                    continue
                
                # 放置皇后
                cols[col] = 1
                diag1[row - col + n - 1] = 1
                diag2[row + col] = 1
                
                # 递归处理下一行
                backtrack(row + 1, cols, diag1, diag2)
                
                # 回溯
                cols[col] = 0
                diag1[row - col + n - 1] = 0
                diag2[row + col] = 0
        
        cols = [0] * n
        diag1 = [0] * (2 * n - 1)  # 主对角线
        diag2 = [0] * (2 * n - 1)  # 副对角线
        
        backtrack(0, cols, diag1, diag2)
        return result
    
    def totalNQueens_optimized(self, n: int) -> int:
        """
        优化解法：位运算 + 预计算
        
        解题思路：
        1. 使用位运算表示每行、每列、每个对角线的占用情况
        2. 使用位运算快速检查冲突
        3. 使用位运算快速更新状态
        4. 使用预计算优化对角线索引
        
        时间复杂度：O(n!)
        空间复杂度：O(1)
        """
        result = 0
        
        def backtrack(row: int, cols: int, diag1: int, diag2: int):
            nonlocal result
            
            if row == n:
                result += 1
                return
            
            # 计算可用的列
            available = ((1 << n) - 1) & (~(cols | diag1 | diag2))
            
            while available:
                # 获取最低位的1
                pos = available & (-available)
                # 清除最低位的1
                available &= (available - 1)
                
                # 递归计算
                backtrack(row + 1, 
                         cols | pos, 
                         (diag1 | pos) << 1, 
                         (diag2 | pos) >> 1)
        
        backtrack(0, 0, 0, 0)
        return result
    
    def totalNQueens_iterative(self, n: int) -> int:
        """
        迭代解法：使用栈模拟递归
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 使用状态保存当前搜索状态
        3. 使用回溯处理冲突
        
        时间复杂度：O(n!)
        空间复杂度：O(n)
        """
        result = 0
        stack = [(0, [0] * n, [0] * (2 * n - 1), [0] * (2 * n - 1))]  # (row, cols, diag1, diag2)
        
        while stack:
            row, cols, diag1, diag2 = stack.pop()
            
            if row == n:
                result += 1
                continue
            
            for col in range(n):
                # 检查冲突
                if (cols[col] or 
                    diag1[row - col + n - 1] or 
                    diag2[row + col]):
                    continue
                
                # 创建新的状态
                new_cols = cols[:]
                new_diag1 = diag1[:]
                new_diag2 = diag2[:]
                
                new_cols[col] = 1
                new_diag1[row - col + n - 1] = 1
                new_diag2[row + col] = 1
                
                stack.append((row + 1, new_cols, new_diag1, new_diag2))
        
        return result
    
    def totalNQueens_memo(self, n: int) -> int:
        """
        记忆化解法：使用记忆化避免重复计算
        
        解题思路：
        1. 使用记忆化存储已计算的结果
        2. 避免重复计算相同的子问题
        3. 提高算法效率
        
        时间复杂度：O(n!)
        空间复杂度：O(n!)
        """
        memo = {}
        
        def backtrack(row: int, cols: int, diag1: int, diag2: int) -> int:
            if row == n:
                return 1
            
            # 检查记忆化
            key = (row, cols, diag1, diag2)
            if key in memo:
                return memo[key]
            
            count = 0
            available = ((1 << n) - 1) & (~(cols | diag1 | diag2))
            
            while available:
                pos = available & (-available)
                available &= (available - 1)
                
                count += backtrack(row + 1, 
                                 cols | pos, 
                                 (diag1 | pos) << 1, 
                                 (diag2 | pos) >> 1)
            
            memo[key] = count
            return count
        
        return backtrack(0, 0, 0, 0)
    
    def totalNQueens_precomputed(self, n: int) -> int:
        """
        预计算解法：使用预计算的结果
        
        解题思路：
        1. 预计算所有可能的N皇后解的数量
        2. 使用查找表快速返回结果
        3. 适用于小规模问题
        
        时间复杂度：O(1)
        空间复杂度：O(1)
        """
        # 预计算的N皇后解的数量
        precomputed = {
            1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352
        }
        
        if n in precomputed:
            return precomputed[n]
        
        # 如果不在预计算范围内，使用标准解法
        return self.totalNQueens(n)


def main():
    """测试标准答案"""
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
    
    # 测试优化解法
    print("测试优化解法...")
    n = 4
    result_opt = solution.totalNQueens_optimized(n)
    assert result_opt == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    n = 4
    result_iter = solution.totalNQueens_iterative(n)
    assert result_iter == expected
    
    # 测试记忆化解法
    print("测试记忆化解法...")
    n = 4
    result_memo = solution.totalNQueens_memo(n)
    assert result_memo == expected
    
    # 测试预计算解法
    print("测试预计算解法...")
    n = 4
    result_pre = solution.totalNQueens_precomputed(n)
    assert result_pre == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
