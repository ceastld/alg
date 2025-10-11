"""
205. 同构字符串
给定两个字符串 s 和 t，判断它们是否是同构的。

如果 s 中的字符可以按某种映射关系替换得到 t，那么这两个字符串是同构的。

每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上。

题目链接：https://leetcode.cn/problems/isomorphic-strings/

示例 1:
输入: s = "egg", t = "add"
输出: true

示例 2:
输入: s = "foo", t = "bar"
输出: false

示例 3:
输入: s = "paper", t = "title"
输出: true

提示：
- 1 <= s.length <= 5 * 10^4
- t.length == s.length
- s 和 t 由任意有效的 ASCII 字符组成
"""

from typing import List


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        ids,idt = {},{}
        for i in range(len(s)):
            cs,ct = s[i],t[i]
            f1,f2 = cs in ids,ct in idt
            if f1 and f2:
                if ids[cs] != idt[ct]:
                    return False
            elif not f1 and not f2:
                ids[cs] = i
                idt[ct] = i
            else:
                return False
        return True
                
            
def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    s = "egg"
    t = "add"
    assert solution.isIsomorphic(s, t) == True

    # 测试用例2
    s = "foo"
    t = "bar"
    assert solution.isIsomorphic(s, t) == False

    # 测试用例3
    s = "paper"
    t = "title"
    assert solution.isIsomorphic(s, t) == True

    # 测试用例4
    s = "badc"
    t = "baba"
    assert solution.isIsomorphic(s, t) == False

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
