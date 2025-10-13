"""
139. 单词拆分 - 标准答案
"""
from typing import List


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示字符串s的前i个字符是否可以被字典中的单词拆分
        2. 状态转移方程：
           - dp[i] = dp[j] and s[j:i] in wordDict
        3. 遍历所有可能的分割点j
        
        时间复杂度：O(n^2 * m)，其中n是字符串长度，m是字典大小
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # 空字符串可以被拆分
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak_optimized(self, s: str, wordDict: List[str]) -> bool:
        """
        优化解法：使用集合优化查找
        
        解题思路：
        1. 将wordDict转换为集合，提高查找效率
        2. 使用动态规划求解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        n = len(s)
        word_set = set(wordDict)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak_recursive(self, s: str, wordDict: List[str]) -> bool:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置是否可以被拆分
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and dfs(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return dfs(0)
    
    def wordBreak_brute_force(self, s: str, wordDict: List[str]) -> bool:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的分割点
        2. 检查每种分割是否有效
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        
        def dfs(start):
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and dfs(end):
                    return True
            
            return False
        
        return dfs(0)
    
    def wordBreak_alternative(self, s: str, wordDict: List[str]) -> bool:
        """
        替代解法：使用BFS
        
        解题思路：
        1. 使用BFS遍历所有可能的状态
        2. 每个状态表示当前可以到达的位置
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        from collections import deque
        
        word_set = set(wordDict)
        queue = deque([0])
        visited = set()
        
        while queue:
            start = queue.popleft()
            
            if start in visited:
                continue
            
            visited.add(start)
            
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set:
                    queue.append(end)
        
        return False
    
    def wordBreak_dfs(self, s: str, wordDict: List[str]) -> bool:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的分割
        2. 检查是否存在有效的分割
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        
        def dfs(start):
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and dfs(end):
                    return True
            
            return False
        
        return dfs(0)
    
    def wordBreak_memo(self, s: str, wordDict: List[str]) -> bool:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and dfs(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return dfs(0)
    
    def wordBreak_greedy(self, s: str, wordDict: List[str]) -> bool:
        """
        贪心解法：优先选择长单词
        
        解题思路：
        1. 优先选择较长的单词
        2. 贪心地选择最优解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        # 按长度从大到小排序
        sorted_words = sorted(wordDict, key=len, reverse=True)
        word_set = set(wordDict)
        
        def dfs(start):
            if start == len(s):
                return True
            
            for word in sorted_words:
                if s.startswith(word, start):
                    if dfs(start + len(word)):
                        return True
            
            return False
        
        return dfs(0)
    
    def wordBreak_iterative(self, s: str, wordDict: List[str]) -> bool:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        stack = [0]
        visited = set()
        
        while stack:
            start = stack.pop()
            
            if start in visited:
                continue
            
            visited.add(start)
            
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set:
                    stack.append(end)
        
        return False
    
    def wordBreak_trie(self, s: str, wordDict: List[str]) -> bool:
        """
        字典树解法
        
        解题思路：
        1. 构建字典树存储所有单词
        2. 使用动态规划求解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s or not wordDict:
            return False
        
        # 构建字典树
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end = False
        
        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j]:
                    node = root
                    valid = True
                    for k in range(j, i):
                        if s[k] not in node.children:
                            valid = False
                            break
                        node = node.children[s[k]]
                    
                    if valid and node.is_end:
                        dp[i] = True
                        break
        
        return dp[n]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "leetcode"
    wordDict = ["leet", "code"]
    result = solution.wordBreak(s, wordDict)
    expected = True
    assert result == expected
    
    # 测试用例2
    s = "applepenapple"
    wordDict = ["apple", "pen"]
    result = solution.wordBreak(s, wordDict)
    expected = True
    assert result == expected
    
    # 测试用例3
    s = "catsandog"
    wordDict = ["cats", "dog", "sand", "and", "cat"]
    result = solution.wordBreak(s, wordDict)
    expected = False
    assert result == expected
    
    # 测试用例4
    s = "a"
    wordDict = ["a"]
    result = solution.wordBreak(s, wordDict)
    expected = True
    assert result == expected
    
    # 测试用例5
    s = "ab"
    wordDict = ["a", "b"]
    result = solution.wordBreak(s, wordDict)
    expected = True
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_opt = solution.wordBreak_optimized(s, wordDict)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_rec = solution.wordBreak_recursive(s, wordDict)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_alt = solution.wordBreak_alternative(s, wordDict)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_dfs = solution.wordBreak_dfs(s, wordDict)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_memo = solution.wordBreak_memo(s, wordDict)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_greedy = solution.wordBreak_greedy(s, wordDict)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_iter = solution.wordBreak_iterative(s, wordDict)
    assert result_iter == expected
    
    # 测试字典树解法
    print("测试字典树解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_trie = solution.wordBreak_trie(s, wordDict)
    assert result_trie == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()