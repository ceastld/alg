"""
344. 反转字符串
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

题目链接：https://leetcode.cn/problems/reverse-string/

示例 1:
输入: ["h","e","l","l","o"]
输出: ["o","l","l","e","h"]

示例 2:
输入: ["H","a","n","n","a","h"]
输出: ["h","a","n","n","a","H"]

提示：
- 1 <= s.length <= 10^5
- s[i] 都是 ASCII 码表中的可打印字符
"""
from typing import List


class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        for i in range(len(s)//2):
            j = len(s)-1-i
            s[i],s[j] = s[j],s[i]



def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = ["h", "e", "l", "l", "o"]
    solution.reverseString(s)
    assert s == ["o", "l", "l", "e", "h"]
    
    # 测试用例2
    s = ["H", "a", "n", "n", "a", "h"]
    solution.reverseString(s)
    assert s == ["h", "a", "n", "n", "a", "H"]
    
    # 测试用例3
    s = ["a"]
    solution.reverseString(s)
    assert s == ["a"]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
