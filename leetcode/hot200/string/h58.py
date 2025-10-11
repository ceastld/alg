"""
剑指Offer58-II. 左旋转字符串
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

题目链接：https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/

示例 1:
输入: s = "abcdefg", k = 2
输出: "cdefgab"

示例 2:
输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"

提示：
- 1 <= k < s.length <= 10000
"""
from typing import List


class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        return s[n:] + s[:n]


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "abcdefg"
    n = 2
    assert solution.reverseLeftWords(s, n) == "cdefgab"
    
    # 测试用例2
    s = "lrloseumgh"
    n = 6
    assert solution.reverseLeftWords(s, n) == "umghlrlose"
    
    # 测试用例3
    s = "hello"
    n = 1
    assert solution.reverseLeftWords(s, n) == "elloh"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
