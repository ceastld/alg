"""
925. 长按键入 - 标准答案
"""
from typing import List


class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        """
        标准解法：双指针法
        
        解题思路：
        1. 使用双指针分别遍历name和typed
        2. 如果字符匹配，两个指针都前进
        3. 如果typed中的字符是重复的，只移动typed指针
        4. 如果字符不匹配，返回False
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        i, j = 0, 0
        
        while i < len(name) and j < len(typed):
            if name[i] == typed[j]:
                i += 1
                j += 1
            elif j > 0 and typed[j] == typed[j - 1]:
                # typed中的字符是重复的，跳过
                j += 1
            else:
                return False
        
        # 检查typed中是否还有未处理的重复字符
        while j < len(typed) and typed[j] == typed[j - 1]:
            j += 1
        
        return i == len(name) and j == len(typed)


def main():
    """测试标准答案"""
    solution = Solution()
    
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
