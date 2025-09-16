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
