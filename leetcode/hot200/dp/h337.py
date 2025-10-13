"""
337. 打家劫舍III
小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。

除了 root 之外，每栋房子有且只有一个"父"房子与之相连。一番侦察之后，聪明的小偷意识到"这个地方的所有房屋的排列类似于一棵二叉树"。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。

给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额。

题目链接：https://leetcode.cn/problems/house-robber-iii/

示例 1:
输入: root = [3,2,3,null,3,null,1]
输出: 7
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7

示例 2:
输入: root = [3,4,5,1,3,null,1]
输出: 9
解释: 小偷一晚能够盗取的最高金额 4 + 1 + 5 = 9

提示：
- 树的节点数在 [1, 10^4] 范围内
- 0 <= Node.val <= 10^4
"""

from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return (0, 0)
            left = dfs(node.left)
            right = dfs(node.right)
            not_rob = max(left) + max(right)
            rob = node.val + left[0] + right[0]
            return (not_rob, rob)
        return max(dfs(root))


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    # 构建二叉树: [3,2,3,null,3,null,1]
    root1 = TreeNode(3)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    root1.left.right = TreeNode(3)
    root1.right.right = TreeNode(1)
    
    result = solution.rob(root1)
    expected = 7
    assert result == expected
    
    # 测试用例2
    # 构建二叉树: [3,4,5,1,3,null,1]
    root2 = TreeNode(3)
    root2.left = TreeNode(4)
    root2.right = TreeNode(5)
    root2.left.left = TreeNode(1)
    root2.left.right = TreeNode(3)
    root2.right.right = TreeNode(1)
    
    result = solution.rob(root2)
    expected = 9
    assert result == expected
    
    # 测试用例3
    # 构建二叉树: [1]
    root3 = TreeNode(1)
    
    result = solution.rob(root3)
    expected = 1
    assert result == expected
    
    # 测试用例4
    # 构建二叉树: [1,2,3]
    root4 = TreeNode(1)
    root4.left = TreeNode(2)
    root4.right = TreeNode(3)
    
    result = solution.rob(root4)
    expected = 5
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
