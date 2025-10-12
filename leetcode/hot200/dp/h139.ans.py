"""
139. 单词拆分 - 标准答案
"""
from typing import List


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示s[0:i]是否可以被字典中的单词拆分
        2. 状态转移方程：dp[i] = dp[j] and s[j:i] in wordDict for j in range(i)
        3. 边界条件：dp[0] = True
        4. 返回dp[len(s)]
        
        时间复杂度：O(n^2 + m)，其中n是s的长度，m是字典中所有单词的总长度
        空间复杂度：O(n + m)
        """
        wordSet = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak_optimized(self, s: str, wordDict: List[str]) -> bool:
        """
        优化解法：限制j的范围
        
        解题思路：
        1. 只检查可能的单词长度
        2. 减少不必要的检查
        
        时间复杂度：O(n×m)，其中m是字典中单词的最大长度
        空间复杂度：O(n + m)
        """
        wordSet = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        # 获取字典中单词的最大长度
        max_len = max(len(word) for word in wordDict) if wordDict else 0
        
        for i in range(1, n + 1):
            # 只检查可能的单词长度
            for j in range(max(0, i - max_len), i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak_recursive(self, s: str, wordDict: List[str]) -> bool:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(n^2 + m)
        空间复杂度：O(n + m)
        """
        wordSet = set(wordDict)
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in wordSet and dfs(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return dfs(0)
    
    def wordBreak_bfs(self, s: str, wordDict: List[str]) -> bool:
        """
        BFS解法：广度优先搜索
        
        解题思路：
        1. 将问题转化为图的最短路径问题
        2. 每个位置是一个节点，单词是边
        3. 使用BFS找到从0到len(s)的路径
        
        时间复杂度：O(n^2 + m)
        空间复杂度：O(n + m)
        """
        from collections import deque
        
        wordSet = set(wordDict)
        n = len(s)
        queue = deque([0])
        visited = set([0])
        
        while queue:
            start = queue.popleft()
            
            if start == n:
                return True
            
            for end in range(start + 1, n + 1):
                if end not in visited and s[start:end] in wordSet:
                    visited.add(end)
                    queue.append(end)
        
        return False
    
    def wordBreak_dfs(self, s: str, wordDict: List[str]) -> bool:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的路径
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n^2 + m)
        空间复杂度：O(n + m)
        """
        wordSet = set(wordDict)
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in wordSet and dfs(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return dfs(0)
    
    def wordBreak_trie(self, s: str, wordDict: List[str]) -> bool:
        """
        字典树解法：使用Trie优化
        
        解题思路：
        1. 构建字典树存储所有单词
        2. 使用动态规划结合字典树
        
        时间复杂度：O(n^2 + m)
        空间复杂度：O(n + m)
        """
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False
        
        # 构建字典树
        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(n):
            if dp[i]:
                node = root
                for j in range(i, n):
                    if s[j] not in node.children:
                        break
                    node = node.children[s[j]]
                    if node.is_word:
                        dp[j + 1] = True
        
        return dp[n]
    
    def wordBreak_optimized_v2(self, s: str, wordDict: List[str]) -> bool:
        """
        进一步优化：使用集合和长度限制
        
        解题思路：
        1. 使用集合存储单词
        2. 只检查可能的单词长度
        3. 提前终止不必要的检查
        
        时间复杂度：O(n×m)
        空间复杂度：O(n + m)
        """
        if not s or not wordDict:
            return False
        
        wordSet = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        # 获取所有可能的单词长度
        word_lengths = set(len(word) for word in wordDict)
        
        for i in range(1, n + 1):
            for length in word_lengths:
                if i >= length and dp[i - length]:
                    if s[i - length:i] in wordSet:
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
    wordDict = ["cat", "cats", "and", "sand", "dog"]
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
    s = "aaaaaaa"
    wordDict = ["aaaa", "aaa"]
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
    
    # 测试BFS解法
    print("测试BFS解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_bfs = solution.wordBreak_bfs(s, wordDict)
    assert result_bfs == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_dfs = solution.wordBreak_dfs(s, wordDict)
    assert result_dfs == expected
    
    # 测试字典树解法
    print("测试字典树解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_trie = solution.wordBreak_trie(s, wordDict)
    assert result_trie == expected
    
    # 测试进一步优化解法
    print("测试进一步优化解法...")
    s = "leetcode"
    wordDict = ["leet", "code"]
    result_opt2 = solution.wordBreak_optimized_v2(s, wordDict)
    assert result_opt2 == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
