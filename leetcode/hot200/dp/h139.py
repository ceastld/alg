"""
139. 单词拆分
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

题目链接：https://leetcode.cn/problems/word-break/

示例 1:
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。

示例 2:
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。

示例 3:
输入: s = "catsandog", wordDict = ["cat", "cats", "and", "sand", "dog"]
输出: false

提示：
- 1 <= s.length <= 300
- 1 <= wordDict.length <= 1000
- 1 <= wordDict[i].length <= 20
- s 和 wordDict[i] 仅有小写英文字母组成
- wordDict 中的所有字符串 互不相同
"""

from typing import List


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
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
        


def main():
    """测试用例"""
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
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
