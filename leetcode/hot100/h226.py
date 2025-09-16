"""
LeetCode 226. Invert Binary Tree

题目描述：
给你一棵二叉树的根节点root，翻转这棵二叉树，并返回其根节点。

示例：
root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]

数据范围：
- 树中节点数目范围在[0, 100]内
- -100 <= Node.val <= 100
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # Base case: if root is None, return None
        if not root:
            return None
        
        # Recursively invert left and right subtrees
        left_inverted = self.invertTree(root.left)
        right_inverted = self.invertTree(root.right)
        
        # Swap left and right children
        root.left = right_inverted
        root.right = left_inverted
        
        return root
