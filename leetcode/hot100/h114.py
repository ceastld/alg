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
