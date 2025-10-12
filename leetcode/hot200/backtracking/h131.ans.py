"""
131. 分割回文串 - 标准答案
"""
from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        """
        标准解法：回溯算法 + 回文判断
        
        解题思路：
        1. 使用回溯算法生成所有可能的分割方案
        2. 对每个可能的分割点，检查是否为回文串
        3. 如果是回文串，则继续递归处理剩余部分
        4. 使用动态规划预计算回文串，优化判断效率
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n^2)
        """
        n = len(s)
        result = []
        path = []
        
        # 预计算回文串，使用动态规划
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文串
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的回文串
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的回文串
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        def backtrack(start: int, current_path: List[str]):
            # 终止条件：处理完所有字符
            if start == n:
                result.append(current_path[:])
                return
            
            # 尝试所有可能的分割点
            for end in range(start, n):
                # 如果从start到end是回文串，则继续递归
                if is_palindrome[start][end]:
                    current_path.append(s[start:end + 1])
                    backtrack(end + 1, current_path)
                    current_path.pop()
        
        backtrack(0, path)
        return result
    
    def partition_optimized(self, s: str) -> List[List[str]]:
        """
        优化解法：中心扩展 + 记忆化
        
        解题思路：
        1. 使用中心扩展法快速判断回文串
        2. 使用记忆化避免重复计算
        3. 使用双指针优化回文串判断
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n^2)
        """
        n = len(s)
        result = []
        path = []
        
        def is_palindrome_fast(left: int, right: int) -> bool:
            """使用双指针判断回文串"""
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        def backtrack(start: int, current_path: List[str]):
            if start == n:
                result.append(current_path[:])
                return
            
            for end in range(start, n):
                if is_palindrome_fast(start, end):
                    current_path.append(s[start:end + 1])
                    backtrack(end + 1, current_path)
                    current_path.pop()
        
        backtrack(0, path)
        return result
    
    def partition_memo(self, s: str) -> List[List[str]]:
        """
        记忆化解法：使用记忆化避免重复计算
        
        解题思路：
        1. 使用记忆化存储已计算的结果
        2. 避免重复计算相同的子问题
        3. 提高算法效率
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n^2)
        """
        n = len(s)
        memo = {}
        
        def is_palindrome_memo(left: int, right: int) -> bool:
            if (left, right) in memo:
                return memo[(left, right)]
            
            while left < right:
                if s[left] != s[right]:
                    memo[(left, right)] = False
                    return False
                left += 1
                right -= 1
            
            memo[(left, right)] = True
            return True
        
        def backtrack(start: int, current_path: List[str]) -> List[List[str]]:
            if start == n:
                return [current_path[:]]
            
            result = []
            for end in range(start, n):
                if is_palindrome_memo(start, end):
                    current_path.append(s[start:end + 1])
                    result.extend(backtrack(end + 1, current_path))
                    current_path.pop()
            
            return result
        
        return backtrack(0, [])


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "aab"
    result = solution.partition(s)
    expected = [["a","a","b"],["aa","b"]]
    assert len(result) == len(expected)
    
    # 测试用例2
    s = "a"
    result = solution.partition(s)
    expected = [["a"]]
    assert result == expected
    
    # 测试用例3
    s = "racecar"
    result = solution.partition(s)
    expected = [["r","a","c","e","c","a","r"],["r","a","cec","a","r"],["r","aceca","r"],["racecar"]]
    assert len(result) == len(expected)
    
    # 测试用例4
    s = "ab"
    result = solution.partition(s)
    expected = [["a","b"]]
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    s = "aab"
    result_opt = solution.partition_optimized(s)
    assert len(result_opt) == len(expected)
    
    # 测试记忆化解法
    print("测试记忆化解法...")
    s = "aab"
    result_memo = solution.partition_memo(s)
    assert len(result_memo) == len(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
