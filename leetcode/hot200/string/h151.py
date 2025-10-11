"""
151. 翻转字符串里的单词
给你一个字符串 s ，逐个翻转字符串中的所有 单词 。

单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。

注意：
- 输入字符串 s 中可能会包含前导空格、尾随空格或者单词间的多个空格。
- 返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

题目链接：https://leetcode.cn/problems/reverse-words-in-a-string/

示例 1:
输入: s = "the sky is blue"
输出: "blue is sky the"

示例 2:
输入: s = "  hello world  "
输出: "world hello"

示例 3:
输入: s = "a good   example"
输出: "example good a"

提示：
- 1 <= s.length <= 10^4
- s 包含英文大小写字母、数字和空格 ' '
- s 中 至少存在一个 单词
"""
from typing import List


class Solution:
    def reverseWords(self, s: str) -> str:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        return " ".join(s.split()[::-1])
        


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "the sky is blue"
    assert solution.reverseWords(s) == "blue is sky the"
    
    # 测试用例2
    s = "  hello world  "
    assert solution.reverseWords(s) == "world hello"
    
    # 测试用例3
    s = "a good   example"
    assert solution.reverseWords(s) == "example good a"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
