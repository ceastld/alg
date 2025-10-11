"""
925. 长按键入
你的朋友正在使用键盘输入他的名字 name。偶尔，在键入字符 c 时，按键可能会被长按，而字符可能被输入 1 次或多次。

你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），那么就返回 True。

题目链接：https://leetcode.cn/problems/long-pressed-name/

示例 1:
输入: name = "alex", typed = "aaleex"
输出: true
解释: 'a' 和 'e' 在 typed 中被长按。

示例 2:
输入: name = "saeed", typed = "ssaaedd"
输出: false
解释: 'e' 一定需要被键入两次，但在 typed 的输出中不是这样。

示例 3:
输入: name = "leelee", typed = "lleeelee"
输出: true

示例 4:
输入: name = "laiden", typed = "laiden"
输出: true
解释: 长按名字中的字符并不是必要的。

提示：
- 1 <= name.length, typed.length <= 1000
- name 和 typed 的字符都是小写字母
"""

from typing import List


class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        """
        请在这里实现你的解法
        """
        i, j = 0, 0
        while i < len(name) and j < len(typed):
            if name[i] == typed[j]:
                i += 1
                j += 1
            elif j > 0 and typed[j] == typed[j - 1]:
                j += 1
            else:
                return False
        while j < len(typed) and typed[j] == typed[j - 1]:
            j += 1
        return i == len(name) and j == len(typed)


def main():
    """测试用例"""
    solution = Solution()
    
    assert solution.isLongPressedName("vtkgn", "vttkgnn") == True
    
    # 测试用例1
    name = "alex"
    typed = "aaleex"
    assert solution.isLongPressedName(name, typed) == True

    # 测试用例2
    name = "saeed"
    typed = "ssaaedd"
    assert solution.isLongPressedName(name, typed) == False

    # 测试用例3
    name = "leelee"
    typed = "lleeelee"
    assert solution.isLongPressedName(name, typed) == True

    # 测试用例4
    name = "laiden"
    typed = "laiden"
    assert solution.isLongPressedName(name, typed) == True

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
