"""
LeetCode 139. Word Break

题目描述：
给你一个字符串s和一个字符串列表wordDict作为字典。请你判断是否可以利用字典中出现的单词拼接出s。
注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

示例：
s = "leetcode", wordDict = ["leet","code"]
输出：true
解释：返回true因为"leetcode"可以由"leet"和"code"拼接成。

数据范围：
- 1 <= s.length <= 300
- 1 <= wordDict.length <= 1000
- 1 <= wordDict[i].length <= 20
- s和wordDict[i]仅有小写英文字母
- wordDict中的所有字符串互不相同
"""

from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        单词拆分 - 动态规划解法
        
        Args:
            s: 目标字符串
            wordDict: 单词字典
            
        Returns:
            是否可以拆分
        """
        if not s or not wordDict:
            return False
        
        # 将字典转换为集合，提高查找效率
        word_set = set(wordDict)
        n = len(s)
        
        # dp[i] 表示 s[0:i] 是否可以拆分
        dp = [False] * (n + 1)
        dp[0] = True  # 空字符串可以拆分
        
        for i in range(1, n + 1):
            for j in range(i):
                # 如果 s[0:j] 可以拆分，且 s[j:i] 在字典中
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreakDFS(self, s: str, wordDict: List[str]) -> bool:
        """
        单词拆分 - DFS + 记忆化搜索
        
        Args:
            s: 目标字符串
            wordDict: 单词字典
            
        Returns:
            是否可以拆分
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        memo = {}  # 记忆化缓存
        
        def dfs(start: int) -> bool:
            if start == len(s):
                return True
            
            if start in memo:
                return memo[start]
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and dfs(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return dfs(0)
    
    def wordBreakBFS(self, s: str, wordDict: List[str]) -> bool:
        """
        单词拆分 - BFS解法
        
        Args:
            s: 目标字符串
            wordDict: 单词字典
            
        Returns:
            是否可以拆分
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        queue = [0]  # 存储可以拆分的位置
        visited = set()  # 避免重复访问
        
        while queue:
            start = queue.pop(0)
            
            if start in visited:
                continue
            visited.add(start)
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set:
                    if end == len(s):
                        return True
                    queue.append(end)
        
        return False
    
    def wordBreakOptimized(self, s: str, wordDict: List[str]) -> bool:
        """
        单词拆分 - 优化版本（剪枝）
        
        Args:
            s: 目标字符串
            wordDict: 单词字典
            
        Returns:
            是否可以拆分
        """
        if not s or not wordDict:
            return False
        
        word_set = set(wordDict)
        max_len = max(len(word) for word in wordDict)  # 最大单词长度
        n = len(s)
        
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            # 只检查可能的单词长度
            for j in range(max(0, i - max_len), i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]


# 测试用例
def test_word_break():
    """测试单词拆分功能"""
    solution = Solution()
    
    # 测试用例1
    s1 = "leetcode"
    wordDict1 = ["leet", "code"]
    result1 = solution.wordBreak(s1, wordDict1)
    print(f"测试1 - s: '{s1}', wordDict: {wordDict1}")
    print(f"结果: {result1}")  # True
    print()
    
    # 测试用例2
    s2 = "applepenapple"
    wordDict2 = ["apple", "pen"]
    result2 = solution.wordBreak(s2, wordDict2)
    print(f"测试2 - s: '{s2}', wordDict: {wordDict2}")
    print(f"结果: {result2}")  # True
    print()
    
    # 测试用例3
    s3 = "catsandog"
    wordDict3 = ["cats", "dog", "sand", "and", "cat"]
    result3 = solution.wordBreak(s3, wordDict3)
    print(f"测试3 - s: '{s3}', wordDict: {wordDict3}")
    print(f"结果: {result3}")  # False
    print()
    
    # 测试用例4
    s4 = "a"
    wordDict4 = ["a"]
    result4 = solution.wordBreak(s4, wordDict4)
    print(f"测试4 - s: '{s4}', wordDict: {wordDict4}")
    print(f"结果: {result4}")  # True
    print()
    
    # 测试用例5
    s5 = "aaaaaaa"
    wordDict5 = ["aaaa", "aaa"]
    result5 = solution.wordBreak(s5, wordDict5)
    print(f"测试5 - s: '{s5}', wordDict: {wordDict5}")
    print(f"结果: {result5}")  # True
    print()
    
    # 对比不同算法
    print("=== 算法对比 ===")
    test_s = "leetcode"
    test_dict = ["leet", "code"]
    
    dp_result = solution.wordBreak(test_s, test_dict)
    dfs_result = solution.wordBreakDFS(test_s, test_dict)
    bfs_result = solution.wordBreakBFS(test_s, test_dict)
    opt_result = solution.wordBreakOptimized(test_s, test_dict)
    
    print(f"动态规划结果: {dp_result}")
    print(f"DFS结果: {dfs_result}")
    print(f"BFS结果: {bfs_result}")
    print(f"优化版本结果: {opt_result}")
    print(f"结果一致: {dp_result == dfs_result == bfs_result == opt_result}")


if __name__ == "__main__":
    test_word_break()