"""
LeetCode 96. Unique Binary Search Trees

题目描述：
给你一个整数n，求恰由n个节点组成且节点值从1到n互不相同的二叉搜索树有多少种？返回满足题意的二叉搜索树的种数。

示例：
n = 3
输出：5

数据范围：
- 1 <= n <= 19
"""

class Solution:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp[i] 表示i个节点能组成的BST数量
        dp = [0] * (n + 1)
        dp[0] = 1  # 空树
        dp[1] = 1  # 只有一个节点
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                # j作为根节点，左子树有j-1个节点，右子树有i-j个节点
                dp[i] += dp[j-1] * dp[i-j]
        
        return dp[n]
