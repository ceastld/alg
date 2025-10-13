"""
968. 监控二叉树
给定一个二叉树，我们在树的节点上安装摄像头。

节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。

计算监控树的所有节点所需的最小摄像头数量。

题目链接：https://leetcode.cn/problems/binary-tree-cameras/

示例 1:
输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

示例 2:
输入：[0,0,null,0,null,0,null,null,0]
输出：2
解释：需要至少两个摄像头来监控树的所有节点。 上图显示了摄像头放置的有效位置之一。

提示：
- 给定树的节点数的范围是 [1, 1000]。
- 每个节点的值都是 0。
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

def main():
    """测试用例"""
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
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
