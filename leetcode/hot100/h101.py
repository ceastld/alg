"""
LeetCode 101. Symmetric Tree

题目描述：
给你一个二叉树的根节点root，检查它是否轴对称。

示例：
root = [1,2,2,3,4,4,3]
输出：true

数据范围：
- 树中节点数目在范围[1, 1000]内
- -100 <= Node.val <= 100
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        
        def is_mirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            
            return is_mirror(left.left, right.right) and is_mirror(left.right, right.left)
        
        return is_mirror(root.left, root.right)
