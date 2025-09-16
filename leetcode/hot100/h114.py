"""
LeetCode 114. Flatten Binary Tree to Linked List

题目描述：
给你二叉树的根结点root，请你将它展开为一个单链表：
- 展开后的单链表应该同样使用TreeNode，其中right子指针指向链表中下一个结点，而left子指针始终为null。
- 展开后的单链表应该与二叉树先序遍历顺序相同。

示例：
root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]

数据范围：
- 树中结点数在范围[0, 2000]内
- -100 <= Node.val <= 100
"""

class Solution:
    def flatten(self, root) -> None:
        def dfs(node):
            if not node:
                return None
            
            # 递归处理左右子树
            left_tail = dfs(node.left)
            right_tail = dfs(node.right)
            
            # 如果左子树存在，将其插入到右子树位置
            if node.left:
                left_tail.right = node.right
                node.right = node.left
                node.left = None
            
            # 返回当前子树的尾节点
            return right_tail or left_tail or node
        
        dfs(root)
