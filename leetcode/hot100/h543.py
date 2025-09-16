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
