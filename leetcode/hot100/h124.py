"""
LeetCode 124. Binary Tree Maximum Path Sum

题目描述：
路径被定义为一条从树中任意节点开始，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中至多出现一次。该路径至少包含一个节点，且不一定经过根节点。
路径和是路径中各节点值的总和。
给你一个二叉树的根节点root，返回其最大路径和。

示例：
root = [1,2,3]
输出：6
解释：最优路径是2->1->3，路径和为2+1+3=6

数据范围：
- 树中节点数目范围是[1, 3 * 10^4]
- -1000 <= Node.val <= 1000
"""

class Solution:
    def maxPathSum(self, root) -> int:
        self.max_sum = float('-inf')
        
        def max_gain(node):
            if not node:
                return 0
            
            # 递归计算左右子树的最大贡献值
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # 当前节点的最大路径和（经过当前节点）
            current_max = node.val + left_gain + right_gain
            
            # 更新全局最大值
            self.max_sum = max(self.max_sum, current_max)
            
            # 返回当前节点的最大贡献值（只能选择一条路径）
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum
