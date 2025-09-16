"""
LeetCode 297. Serialize and Deserialize Binary Tree

题目描述：
序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。
请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列/反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

示例：
root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]

数据范围：
- 树中节点数范围是[0, 10^4]
- -1000 <= Node.val <= 1000
"""

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    def serialize(self, root):
        def preorder(node):
            if not node:
                return "null"
            return str(node.val) + "," + preorder(node.left) + "," + preorder(node.right)
        
        return preorder(root)
    
    def deserialize(self, data):
        def build():
            val = next(vals)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        
        vals = iter(data.split(","))
        return build()
