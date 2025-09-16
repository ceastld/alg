"""
LeetCode 538. Convert BST to Greater Tree

题目描述：
给出二叉搜索树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点node的新值等于原树中大于或等于node.val的值之和。
提醒一下，二叉搜索树满足下列约束条件：
- 节点的左子树仅包含键小于节点键的节点
- 节点的右子树仅包含键大于节点键的节点
- 左右子树也必须是二叉搜索树

示例：
root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

数据范围：
- 树中的节点数介于0和10^4之间
- 每个节点的值介于-10^4和10^4之间
- 树中的所有值互不相同
- 给定的树为二叉搜索树
"""

class Solution:
    def convertBST(self, root) -> int:
        self.sum = 0
        
        def reverse_inorder(node):
            if not node:
                return
            
            reverse_inorder(node.right)
            self.sum += node.val
            node.val = self.sum
            reverse_inorder(node.left)
        
        reverse_inorder(root)
        return root
