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
