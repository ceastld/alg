"""
763. 划分字母区间
字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

题目链接：https://leetcode.cn/problems/partition-labels/

示例 1:
输入：S = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegdehijhklij" 的划分是错误的，因为划分的片段数较少。

示例 2:
输入：S = "eccbbbbdec"
输出：[10]

提示：
- S的长度在[1, 500]之间。
- S只包含小写字母'a'到'z'。
"""

from typing import List


class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8]
    
    # 测试用例2
    assert solution.partitionLabels("eccbbbbdec") == [10]
    
    # 测试用例3
    assert solution.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8]
    
    # 测试用例4
    assert solution.partitionLabels("a") == [1]
    
    # 测试用例5
    assert solution.partitionLabels("ab") == [1,1]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
