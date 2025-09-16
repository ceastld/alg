class Solution:
    def rob(self, root) -> int:
        def dfs(node):
            if not node:
                return (0, 0)  # (不偷, 偷)
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # 当前节点不偷：左右子树都可以偷或不偷
            not_rob = max(left) + max(right)
            
            # 当前节点偷：左右子树都不能偷
            rob = node.val + left[0] + right[0]
            
            return (not_rob, rob)
        
        return max(dfs(root))
