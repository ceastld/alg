"""
LCR122. 路径加密
假定一段路径记作字符串 path，其中以 "." 作为分隔符。现需将路径加密，加密方法为将 path 中的分隔符替换为空格 " "，请返回加密后的字符串。

题目链接：https://leetcode.cn/problems/ti-huan-kong-ge-lcof/

示例 1:
输入: path = "a.aef.qerf.bb"
输出: "a aef qerf bb"

示例 2:
输入: path = "hello.world"
输出: "hello world"

示例 3:
输入: path = "a"
输出: "a"

提示：
- 1 <= path.length <= 10000
- path 由小写英文字母和 '.' 组成
"""
from typing import List


class Solution:
    def pathEncryption(self, path: str) -> str:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    path = "a.aef.qerf.bb"
    assert solution.pathEncryption(path) == "a aef qerf bb"
    
    # 测试用例2
    path = "hello.world"
    assert solution.pathEncryption(path) == "hello world"
    
    # 测试用例3
    path = "a"
    assert solution.pathEncryption(path) == "a"
    
    # 测试用例4
    path = "a.b.c.d"
    assert solution.pathEncryption(path) == "a b c d"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()