"""
LeetCode 98. Validate Binary Search Tree

题目描述：
给你一个二叉树的根节点root，判断其是否是一个有效的二叉搜索树。
有效二叉搜索树定义如下：
- 节点的左子树只包含小于当前节点的数
- 节点的右子树只包含大于当前节点的数
- 所有左子树和右子树自身必须也是二叉搜索树

示例：
root = [2,1,3]
输出：true

数据范围：
- 树中节点数目范围在[1, 10^4]内
- -2^31 <= Node.val <= 2^31 - 1
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def validate(node, min_val, max_val):
            if not node:
                return True
            
            if node.val <= min_val or node.val >= max_val:
                return False
            
            return (validate(node.left, min_val, node.val) and 
                    validate(node.right, node.val, max_val))
        
        return validate(root, float('-inf'), float('inf'))
