"""
72. 编辑距离
给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

题目链接：https://leetcode.cn/problems/edit-distance/

示例 1:
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')

示例 2:
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

提示：
- 0 <= word1.length, word2.length <= 500
- word1 和 word2 由小写英文字母组成
"""

from typing import List


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    word1 = "horse"
    word2 = "ros"
    result = solution.minDistance(word1, word2)
    expected = 3
    assert result == expected
    
    # 测试用例2
    word1 = "intention"
    word2 = "execution"
    result = solution.minDistance(word1, word2)
    expected = 5
    assert result == expected
    
    # 测试用例3
    word1 = ""
    word2 = ""
    result = solution.minDistance(word1, word2)
    expected = 0
    assert result == expected
    
    # 测试用例4
    word1 = "a"
    word2 = "b"
    result = solution.minDistance(word1, word2)
    expected = 1
    assert result == expected
    
    # 测试用例5
    word1 = "abc"
    word2 = "abc"
    result = solution.minDistance(word1, word2)
    expected = 0
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
