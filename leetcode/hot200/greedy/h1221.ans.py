"""
1221. 分割平衡字符串 - 标准答案
"""
from typing import List

class Solution:
    def balancedStringSplit(self, s: str) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 遍历字符串，维护L和R的计数
        2. 当L和R的计数相等时，说明找到了一个平衡字符串
        3. 立即分割，计数器重置，继续寻找下一个平衡字符串
        4. 贪心策略：一旦找到平衡就立即分割，这样能获得最大数量
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        count = 0
        l_count = 0
        r_count = 0
        
        for char in s:
            if char == 'L':
                l_count += 1
            else:
                r_count += 1
            
            # 当L和R数量相等时，找到一个平衡字符串
            if l_count == r_count:
                count += 1
                l_count = 0
                r_count = 0
        
        return count

def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.balancedStringSplit("RLRRLLRLRL") == 4
    
    # 测试用例2
    assert solution.balancedStringSplit("RLLLLRRRLR") == 3
    
    # 测试用例3
    assert solution.balancedStringSplit("LLLLRRRR") == 1
    
    print("所有测试用例通过！")
