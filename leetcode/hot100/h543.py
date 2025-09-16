"""
LeetCode 543. Diameter of Binary Tree

题目描述：
给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

示例：
root = [1,2,3,4,5]
输出：3
解释：它的长度是路径[4,2,1,3]或者[5,2,1,3]。

数据范围：
- 树中结点数目在范围[1, 10^4]内
- -100 <= Node.val <= 100
"""

class Solution:
    def diameterOfBinaryTree(self, root) -> int:
        self.max_diameter = 0
        
        def depth(node):
            if not node:
                return 0
            
            left_depth = depth(node.left)
            right_depth = depth(node.right)
            
            # 更新最大直径
            self.max_diameter = max(self.max_diameter, left_depth + right_depth)
            
            # 返回当前节点的最大深度
            return max(left_depth, right_depth) + 1
        
        depth(root)
        return self.max_diameter
