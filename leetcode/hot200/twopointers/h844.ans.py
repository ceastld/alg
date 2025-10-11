"""
844. 比较含退格的字符串 - 标准答案
"""
from typing import List


class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        """
        标准解法：双指针法（从后往前）
        
        解题思路：
        1. 从字符串末尾开始遍历
        2. 遇到'#'时，退格计数器+1
        3. 遇到普通字符时，如果退格计数器>0，则跳过该字符
        4. 否则比较两个字符串的当前字符
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        def get_next_char(string, index):
            """获取字符串中下一个有效字符的索引"""
            backspace_count = 0
            while index >= 0:
                if string[index] == '#':
                    backspace_count += 1
                elif backspace_count > 0:
                    backspace_count -= 1
                else:
                    return index
                index -= 1
            return -1
        
        i, j = len(s) - 1, len(t) - 1
        
        while i >= 0 or j >= 0:
            i = get_next_char(s, i)
            j = get_next_char(t, j)
            
            if i < 0 and j < 0:
                return True
            if i < 0 or j < 0:
                return False
            if s[i] != t[j]:
                return False
            
            i -= 1
            j -= 1
        
        return True


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "ab#c"
    t = "ad#c"
    assert solution.backspaceCompare(s, t) == True
    
    # 测试用例2
    s = "ab##"
    t = "c#d#"
    assert solution.backspaceCompare(s, t) == True
    
    # 测试用例3
    s = "a#c"
    t = "b"
    assert solution.backspaceCompare(s, t) == False
    
    # 测试用例4
    s = "a##c"
    t = "#a#c"
    assert solution.backspaceCompare(s, t) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
