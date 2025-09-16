"""
LeetCode 437. Path Sum III

题目描述：
给定一个二叉树的根节点root，和一个整数targetSum，求该二叉树里节点值之和等于targetSum的路径的数目。
路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

示例：
root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于8的路径有3条，如图所示。

数据范围：
- 二叉树的节点个数的范围是[0, 1000]
- -10^9 <= Node.val <= 10^9
- -1000 <= targetSum <= 1000
"""

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
