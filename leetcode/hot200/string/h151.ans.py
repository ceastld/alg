"""
151. 翻转字符串里的单词 - 标准答案
"""
from typing import List


class Solution:
    def reverseWords(self, s: str) -> str:
        """
        标准解法：分割+反转+连接
        
        解题思路：
        1. 使用split()分割字符串，自动处理多个空格
        2. 反转单词列表
        3. 使用join()连接单词，用单个空格分隔
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        # 分割字符串，split()会自动处理多个空格
        words = s.split()
        # 反转单词列表
        words.reverse()
        # 用单个空格连接
        return ' '.join(words)


def main():
    """测试标准答案"""
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
