"""
LeetCode 236. Lowest Common Ancestor of a Binary Tree

题目描述：
给定一个二叉树，找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为："对于有根树T的两个节点p、q，最近公共祖先表示为一个节点x，满足x是p、q的祖先且x的深度尽可能大（一个节点也可以是它自己的祖先）。"

示例：
root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点5和节点1的最近公共祖先是节点3。

数据范围：
- 树中节点数目在范围[2, 10^5]内
- -10^9 <= Node.val <= 10^9
- 所有Node.val互不相同
- p != q
- p和q均存在于给定的二叉树中
"""

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is not None and right is not None:
            return root
        return left if left is not None else right