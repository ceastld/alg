"""
968. 监控二叉树 - 标准答案
"""
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        """
        标准解法：贪心算法（后序遍历）
        
        解题思路：
        1. 使用后序遍历，从叶子节点开始
        2. 定义三种状态：0-未覆盖，1-已覆盖，2-有摄像头
        3. 贪心策略：优先在叶子节点的父节点放置摄像头
        
        时间复杂度：O(n)
        空间复杂度：O(h)，h为树的高度
        """
        self.result = 0
        
        def dfs(node):
            if not node:
                return 1  # 空节点视为已覆盖
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # 如果左右子节点都未覆盖，当前节点需要放置摄像头
            if left == 0 or right == 0:
                self.result += 1
                return 2  # 当前节点有摄像头
            
            # 如果左右子节点有摄像头，当前节点已覆盖
            if left == 2 or right == 2:
                return 1  # 当前节点已覆盖
            
            # 否则，当前节点未覆盖
            return 0
        
        # 如果根节点未覆盖，需要放置摄像头
        if dfs(root) == 0:
            self.result += 1
        
        return self.result
    
    def minCameraCover_alternative(self, root: Optional[TreeNode]) -> int:
        """
        替代解法：贪心算法（使用字典）
        
        解题思路：
        1. 使用字典记录每个节点的状态
        2. 后序遍历，从叶子节点开始
        3. 贪心策略：优先在叶子节点的父节点放置摄像头
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        self.result = 0
        covered = set()
        
        def dfs(node, parent):
            if not node:
                return
            
            dfs(node.left, node)
            dfs(node.right, node)
            
            # 如果当前节点未覆盖且不是根节点，在父节点放置摄像头
            if (not node.left or node.left in covered) and (not node.right or node.right in covered):
                if parent:
                    covered.add(parent)
                    covered.add(node)
                    self.result += 1
                elif node not in covered:
                    covered.add(node)
                    self.result += 1
        
        dfs(root, None)
        return self.result
    
    def minCameraCover_optimized(self, root: Optional[TreeNode]) -> int:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 使用后序遍历，从叶子节点开始
        2. 定义三种状态：0-未覆盖，1-已覆盖，2-有摄像头
        3. 贪心策略：优先在叶子节点的父节点放置摄像头
        
        时间复杂度：O(n)
        空间复杂度：O(h)，h为树的高度
        """
        self.result = 0
        
        def dfs(node):
            if not node:
                return 1
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            if left == 0 or right == 0:
                self.result += 1
                return 2
            
            if left == 2 or right == 2:
                return 1
            
            return 0
        
        if dfs(root) == 0:
            self.result += 1
        
        return self.result
    
    def minCameraCover_detailed(self, root: Optional[TreeNode]) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 使用后序遍历，从叶子节点开始
        2. 定义三种状态：0-未覆盖，1-已覆盖，2-有摄像头
        3. 贪心策略：优先在叶子节点的父节点放置摄像头
        
        时间复杂度：O(n)
        空间复杂度：O(h)，h为树的高度
        """
        self.result = 0
        
        def dfs(node):
            if not node:
                return 1  # 空节点视为已覆盖
            
            # 后序遍历：先处理左右子节点
            left = dfs(node.left)
            right = dfs(node.right)
            
            # 如果左右子节点都未覆盖，当前节点需要放置摄像头
            if left == 0 or right == 0:
                self.result += 1
                return 2  # 当前节点有摄像头
            
            # 如果左右子节点有摄像头，当前节点已覆盖
            if left == 2 or right == 2:
                return 1  # 当前节点已覆盖
            
            # 否则，当前节点未覆盖
            return 0
        
        # 如果根节点未覆盖，需要放置摄像头
        if dfs(root) == 0:
            self.result += 1
        
        return self.result
    
    def minCameraCover_brute_force(self, root: Optional[TreeNode]) -> int:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的摄像头放置方案
        2. 检查每个方案是否能覆盖所有节点
        3. 返回最少的摄像头数量
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        def is_covered(node, cameras):
            if not node:
                return True
            
            # 检查当前节点是否被摄像头覆盖
            if node in cameras:
                return True
            
            # 检查父节点是否有摄像头
            if hasattr(node, 'parent') and node.parent in cameras:
                return True
            
            # 检查子节点是否有摄像头
            if node.left in cameras or node.right in cameras:
                return True
            
            return False
        
        def backtrack(node, cameras):
            if not node:
                return len(cameras)
            
            # 不放置摄像头
            result = backtrack(node.left, cameras) + backtrack(node.right, cameras)
            
            # 放置摄像头
            cameras.add(node)
            result = min(result, backtrack(node.left, cameras) + backtrack(node.right, cameras))
            cameras.remove(node)
            
            return result
        
        return backtrack(root, set())
    
    def minCameraCover_step_by_step(self, root: Optional[TreeNode]) -> int:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：后序遍历，从叶子节点开始
        2. 第二步：贪心选择摄像头位置
        
        时间复杂度：O(n)
        空间复杂度：O(h)，h为树的高度
        """
        self.result = 0
        
        def dfs(node):
            if not node:
                return 1
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            if left == 0 or right == 0:
                self.result += 1
                return 2
            
            if left == 2 or right == 2:
                return 1
            
            return 0
        
        if dfs(root) == 0:
            self.result += 1
        
        return self.result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    # 构建树：[0,0,null,0,0]
    root1 = TreeNode(0)
    root1.left = TreeNode(0)
    root1.left.left = TreeNode(0)
    root1.left.right = TreeNode(0)
    assert solution.minCameraCover(root1) == 1
    
    # 测试用例2
    # 构建树：[0,0,null,0,null,0,null,null,0]
    root2 = TreeNode(0)
    root2.left = TreeNode(0)
    root2.left.left = TreeNode(0)
    root2.left.left.left = TreeNode(0)
    root2.left.left.left.right = TreeNode(0)
    assert solution.minCameraCover(root2) == 2
    
    # 测试用例3
    # 构建树：[0]
    root3 = TreeNode(0)
    assert solution.minCameraCover(root3) == 1
    
    # 测试用例4：边界情况
    # 构建树：[0,0,0]
    root4 = TreeNode(0)
    root4.left = TreeNode(0)
    root4.right = TreeNode(0)
    assert solution.minCameraCover(root4) == 1
    
    # 测试用例5：复杂情况
    # 构建树：[0,0,null,0,0]
    root5 = TreeNode(0)
    root5.left = TreeNode(0)
    root5.left.left = TreeNode(0)
    root5.left.right = TreeNode(0)
    assert solution.minCameraCover(root5) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
