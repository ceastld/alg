"""
337. 打家劫舍III - 标准答案
"""
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        """
        标准解法：动态规划（树形DP）
        
        解题思路：
        1. 定义状态：每个节点返回两个值 [不偷, 偷]
        2. 状态转移：
           - 不偷当前节点：max(左子树不偷, 左子树偷) + max(右子树不偷, 右子树偷)
           - 偷当前节点：左子树不偷 + 右子树不偷 + 当前节点值
        3. 返回根节点的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(h)，h为树的高度
        """
        if not root:
            return 0
        
        def dfs(node):
            if not node:
                return [0, 0]  # [不偷, 偷]
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # 不偷当前节点：左右子树可以偷或不偷
            not_rob = max(left[0], left[1]) + max(right[0], right[1])
            # 偷当前节点：左右子树都不能偷
            rob = left[0] + right[0] + node.val
            
            return [not_rob, rob]
        
        result = dfs(root)
        return max(result[0], result[1])
    
    def rob_recursive(self, root: Optional[TreeNode]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个节点的最大利润
        2. 使用记忆化避免重复计算
        3. 考虑父子节点不能同时被偷
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not root:
            return 0
        
        memo = {}
        
        def dfs(node, can_rob):
            if not node:
                return 0
            
            if (node, can_rob) in memo:
                return memo[(node, can_rob)]
            
            if can_rob:
                # 可以选择偷或不偷
                result = max(dfs(node.left, False) + dfs(node.right, False) + node.val,
                           dfs(node.left, True) + dfs(node.right, True))
            else:
                # 不能偷，只能从子节点获取
                result = dfs(node.left, True) + dfs(node.right, True)
            
            memo[(node, can_rob)] = result
            return result
        
        return dfs(root, True)
    
    def rob_brute_force(self, root: Optional[TreeNode]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的偷窃方案
        2. 计算每种方案的利润
        3. 返回最大利润
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not root:
            return 0
        
        def dfs(node, can_rob):
            if not node:
                return 0
            
            if can_rob:
                # 可以选择偷或不偷
                return max(dfs(node.left, False) + dfs(node.right, False) + node.val,
                          dfs(node.left, True) + dfs(node.right, True))
            else:
                # 不能偷，只能从子节点获取
                return dfs(node.left, True) + dfs(node.right, True)
        
        return dfs(root, True)
    
    def rob_optimized(self, root: Optional[TreeNode]) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用后序遍历，从叶子节点开始计算
        2. 每个节点只计算一次
        3. 避免重复计算
        
        时间复杂度：O(n)
        空间复杂度：O(h)
        """
        if not root:
            return 0
        
        def dfs(node):
            if not node:
                return [0, 0]  # [不偷, 偷]
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # 不偷当前节点：左右子树可以偷或不偷
            not_rob = max(left[0], left[1]) + max(right[0], right[1])
            # 偷当前节点：左右子树都不能偷
            rob = left[0] + right[0] + node.val
            
            return [not_rob, rob]
        
        result = dfs(root)
        return max(result[0], result[1])
    
    def rob_alternative(self, root: Optional[TreeNode]) -> int:
        """
        替代解法：使用两个状态
        
        解题思路：
        1. 定义两个状态：偷当前节点、不偷当前节点
        2. 状态转移：
           - 偷当前节点：左子树不偷 + 右子树不偷 + 当前节点值
           - 不偷当前节点：max(左子树偷, 左子树不偷) + max(右子树偷, 右子树不偷)
        
        时间复杂度：O(n)
        空间复杂度：O(h)
        """
        if not root:
            return 0
        
        def dfs(node):
            if not node:
                return [0, 0]  # [不偷, 偷]
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # 不偷当前节点：左右子树可以偷或不偷
            not_rob = max(left[0], left[1]) + max(right[0], right[1])
            # 偷当前节点：左右子树都不能偷
            rob = left[0] + right[0] + node.val
            
            return [not_rob, rob]
        
        result = dfs(root)
        return max(result[0], result[1])
    
    def rob_iterative(self, root: Optional[TreeNode]) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 后序遍历计算每个节点的状态
        3. 避免递归调用栈
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not root:
            return 0
        
        stack = []
        visited = set()
        result = {}
        
        stack.append(root)
        
        while stack:
            node = stack[-1]
            
            if node in visited:
                stack.pop()
                
                # 计算当前节点的状态
                left = result.get(node.left, [0, 0])
                right = result.get(node.right, [0, 0])
                
                # 不偷当前节点：左右子树可以偷或不偷
                not_rob = max(left[0], left[1]) + max(right[0], right[1])
                # 偷当前节点：左右子树都不能偷
                rob = left[0] + right[0] + node.val
                
                result[node] = [not_rob, rob]
            else:
                visited.add(node)
                
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        
        final_result = result[root]
        return max(final_result[0], final_result[1])


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    # 构建二叉树: [3,2,3,null,3,null,1]
    root1 = TreeNode(3)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    root1.left.right = TreeNode(3)
    root1.right.right = TreeNode(1)
    
    result = solution.rob(root1)
    expected = 7
    assert result == expected
    
    # 测试用例2
    # 构建二叉树: [3,4,5,1,3,null,1]
    root2 = TreeNode(3)
    root2.left = TreeNode(4)
    root2.right = TreeNode(5)
    root2.left.left = TreeNode(1)
    root2.left.right = TreeNode(3)
    root2.right.right = TreeNode(1)
    
    result = solution.rob(root2)
    expected = 9
    assert result == expected
    
    # 测试用例3
    # 构建二叉树: [1]
    root3 = TreeNode(1)
    
    result = solution.rob(root3)
    expected = 1
    assert result == expected
    
    # 测试用例4
    # 构建二叉树: [1,2,3]
    root4 = TreeNode(1)
    root4.left = TreeNode(2)
    root4.right = TreeNode(3)
    
    result = solution.rob(root4)
    expected = 5
    assert result == expected
    
    # 测试递归解法
    print("测试递归解法...")
    root1 = TreeNode(3)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    root1.left.right = TreeNode(3)
    root1.right.right = TreeNode(1)
    
    result_rec = solution.rob_recursive(root1)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    root1 = TreeNode(3)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    root1.left.right = TreeNode(3)
    root1.right.right = TreeNode(1)
    
    result_opt = solution.rob_optimized(root1)
    assert result_opt == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    root1 = TreeNode(3)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    root1.left.right = TreeNode(3)
    root1.right.right = TreeNode(1)
    
    result_iter = solution.rob_iterative(root1)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
