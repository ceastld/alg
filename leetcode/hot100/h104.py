"""
LeetCode 104. Maximum Depth of Binary Tree

题目描述：
给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
说明：叶子节点是指没有子节点的节点。

示例：
root = [3,9,20,null,null,15,7]
输出：3

数据范围：
- 树中节点的数量在[0, 10^4]范围内
- -100 <= Node.val <= 100
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        # 递归计算左右子树的最大深度
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        
        # 返回左右子树最大深度 + 1（当前节点）
        return max(left_depth, right_depth) + 1
