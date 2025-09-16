class Solution:
    def pathSum(self, root, targetSum: int) -> int:
        from collections import defaultdict
        
        def dfs(node, current_sum):
            if not node:
                return 0
            
            current_sum += node.val
            count = prefix_sum[current_sum - targetSum]
            
            prefix_sum[current_sum] += 1
            
            count += dfs(node.left, current_sum)
            count += dfs(node.right, current_sum)
            
            prefix_sum[current_sum] -= 1
            
            return count
        
        prefix_sum = defaultdict(int)
        prefix_sum[0] = 1
        
        return dfs(root, 0)
