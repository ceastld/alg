"""
657. 机器人能否返回原点 - 标准答案
"""
from typing import List


class Solution:
    def judgeCircle(self, moves: str) -> bool:
        """
        标准解法：计数法
        
        解题思路：
        1. 统计上下左右四个方向的移动次数
        2. 如果向上的次数等于向下的次数，且向左的次数等于向右的次数，则返回原点
        3. 否则没有返回原点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        x, y = 0, 0
        
        for move in moves:
            if move == 'U':
                y += 1
            elif move == 'D':
                y -= 1
            elif move == 'L':
                x -= 1
            elif move == 'R':
                x += 1
        
        return x == 0 and y == 0


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    moves = "UD"
    assert solution.judgeCircle(moves) == True
    
    # 测试用例2
    moves = "LL"
    assert solution.judgeCircle(moves) == False
    
    # 测试用例3
    moves = "UDLR"
    assert solution.judgeCircle(moves) == True
    
    # 测试用例4
    moves = "UUDD"
    assert solution.judgeCircle(moves) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
