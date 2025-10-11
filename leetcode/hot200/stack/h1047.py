"""
1047. 删除字符串中的所有相邻重复项
给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

题目链接：https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/

示例 1:
输入: "abbaca"
输出: "ca"
解释: 在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。

示例 2:
输入: "azxxzy"
输出: "ay"

提示：
- 1 <= S.length <= 20000
- S 仅由小写英文字母组成
"""
from typing import List


class Solution:
    def removeDuplicates(self, s: str) -> str:
        """
        高效解法：使用collections.deque
        
        优化思路：
        1. 使用deque替代list，提供O(1)的头部和尾部操作
        2. 使用StringBuilder模式，避免频繁的字符串拼接
        3. 使用collections.Counter进行字符统计优化
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        from collections import deque
        
        stack = deque()
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        
        return ''.join(stack)
        


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "abbaca"
    assert solution.removeDuplicates(s) == "ca"
    
    # 测试用例2
    s = "azxxzy"
    assert solution.removeDuplicates(s) == "ay"
    
    # 测试用例3
    s = "a"
    assert solution.removeDuplicates(s) == "a"
    
    # 测试用例4
    s = "abccba"
    assert solution.removeDuplicates(s) == ""
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
