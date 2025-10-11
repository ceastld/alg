"""
657. 机器人能否返回原点
在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，判断这个机器人在完成移动后是否在 (0, 0) 处。

移动顺序由字符串 moves 表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），U（上）和 D（下）。

如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。

注意：机器人"面朝"的方向无关紧要。 "R" 将始终使机器人向右移动一次，"L" 将始终向左移动一次，等等。此外，假设每次移动机器人的移动幅度相同。

题目链接：https://leetcode.cn/problems/robot-return-to-origin/

示例 1:
输入: moves = "UD"
输出: true
解释: 机器人向上移动一次，然后向下移动一次。所有动作都具有相同的幅度，因此它最终回到它开始的原点。因此，我们返回 true。

示例 2:
输入: moves = "LL"
输出: false
解释: 机器人向左移动两次。它最终位于原点的左侧，距原点有两次移动的距离。我们返回 false。

提示：
- 1 <= moves.length <= 2 * 10^4
- moves 只包含字符 'U', 'D', 'L', 'R'
"""
from collections import Counter
from typing import List


class Solution:
    def judgeCircle(self, moves: str) -> bool:
        """
        请在这里实现你的解法
        """
        count = Counter(moves)
        return count['U'] == count['D'] and count['L'] == count['R']


def main():
    """测试用例"""
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
