"""
LeetCode 105. Construct Binary Tree from Preorder and Inorder Traversal

题目描述：
给定两个整数数组preorder和inorder，其中preorder是二叉树的先序遍历，inorder是同一棵树的中序遍历，请构造二叉树并返回其根节点。

示例：
preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出：[3,9,20,null,null,15,7]

数据范围：
- 1 <= preorder.length <= 3000
- inorder.length == preorder.length
- -3000 <= preorder[i], inorder[i] <= 3000
- preorder和inorder均无重复元素
- inorder均出现在preorder中
- preorder保证为二叉树的前序遍历序列
- inorder保证为二叉树的中序遍历序列
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder or not inorder:
            return None
        
        # 前序遍历的第一个元素是根节点
        root_val = preorder[0]
        root = TreeNode(root_val)
        
        # 在中序遍历中找到根节点的位置
        root_index = inorder.index(root_val)
        
        # 递归构建左右子树
        root.left = self.buildTree(preorder[1:root_index+1], inorder[:root_index])
        root.right = self.buildTree(preorder[root_index+1:], inorder[root_index+1:])
        
        return root
