"""
131. 分割回文串
给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

回文串 是正着读和反着读都一样的字符串。

题目链接：https://leetcode.cn/problems/palindrome-partitioning/

示例 1:
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]

示例 2:
输入：s = "a"
输出：[["a"]]

提示：
- 1 <= s.length <= 16
- s 仅由小写英文字母组成
"""

from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:

        def dfs(start, path):
            if start == len(s):
                result.append(path.copy())
                return
            for end in range(start + 1, len(s) + 1):
                i, j = start, end - 1
                while i < j and s[i] == s[j]:
                    i += 1
                    j -= 1
                if i >= j:
                    path.append(s[start:end])
                    dfs(end, path)
                    path.pop()
        result = []
        dfs(0, [])
        return result

def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    s = "aab"
    result = solution.partition(s)
    expected = [["a", "a", "b"], ["aa", "b"]]
    assert len(result) == len(expected)

    # 测试用例2
    s = "a"
    result = solution.partition(s)
    expected = [["a"]]
    assert result == expected

    # 测试用例3
    s = "racecar"
    result = solution.partition(s)
    expected = [["r", "a", "c", "e", "c", "a", "r"], ["r", "a", "cec", "a", "r"], ["r", "aceca", "r"], ["racecar"]]
    assert len(result) == len(expected)

    # 测试用例4
    s = "ab"
    result = solution.partition(s)
    expected = [["a", "b"]]
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
