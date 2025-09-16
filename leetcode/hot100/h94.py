"""
LeetCode 94. Binary Tree Inorder Traversal

题目描述：
给定一个二叉树的根节点root，返回它的中序遍历。

示例：
root = [1,null,2,3]
输出：[1,3,2]

数据范围：
- 树中节点数目在范围[0, 100]内
- -100 <= Node.val <= 100
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        
        def inorder(node):
            if not node:
                return
            
            inorder(node.left)   # 左
            result.append(node.val)  # 根
            inorder(node.right)  # 右
        
        inorder(root)
        return result
