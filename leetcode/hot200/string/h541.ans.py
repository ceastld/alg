"""
541. 反转字符串II - 标准答案
"""
from typing import List


class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        """
        标准解法：分段处理
        
        解题思路：
        1. 每2k个字符为一段
        2. 对每段的前k个字符进行反转
        3. 如果剩余字符少于k个，全部反转
        4. 如果剩余字符在k到2k之间，只反转前k个
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        s_list = list(s)
        n = len(s_list)
        
        for i in range(0, n, 2 * k):
            # 确定反转的结束位置
            end = min(i + k, n)
            # 反转前k个字符
            s_list[i:end] = s_list[i:end][::-1]
        
        return ''.join(s_list)


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "abcdefg"
    k = 2
    assert solution.reverseStr(s, k) == "bacdfeg"
    
    # 测试用例2
    s = "abcd"
    k = 2
    assert solution.reverseStr(s, k) == "bacd"
    
    # 测试用例3
    s = "a"
    k = 1
    assert solution.reverseStr(s, k) == "a"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
