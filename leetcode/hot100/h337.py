"""
LeetCode 337. House Robber III

题目描述：
小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为root。
除了root之外，每栋房子有且只有一个"父"房子与之相连。一番侦察之后，聪明的小偷意识到"这个地方的所有房屋的排列类似于一棵二叉树"。如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
给定二叉树的root。返回在不触动警报的情况下，小偷能够盗取的最高金额。

示例：
root = [3,2,3,null,3,null,1]
输出：7
解释：小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7

数据范围：
- 树的节点数在[1, 10^4]范围内
- 0 <= Node.val <= 10^4
"""

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
