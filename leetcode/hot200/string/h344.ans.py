"""
344. 反转字符串 - 标准答案
"""
from typing import List


class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        标准解法：双指针
        
        解题思路：
        1. 使用左右两个指针
        2. 交换指针位置的字符
        3. 指针向中间移动
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1


def main():
    """测试标准答案"""
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
