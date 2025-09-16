"""
LeetCode 102. Binary Tree Level Order Traversal

题目描述：
给你二叉树的根节点root，返回其节点值的层序遍历。（即逐层地，从左到右访问所有节点）。

示例：
root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]

数据范围：
- 树中节点数目在范围[0, 2000]内
- -1000 <= Node.val <= 1000
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        result = []
        queue = [root]
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            # 处理当前层的所有节点
            for _ in range(level_size):
                node = queue.pop(0)
                level_values.append(node.val)
                
                # 将子节点加入队列
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level_values)
        
        return result
