"""
141. 环形链表
给你一个链表的头节点 head，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递。仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true。否则，返回 false。

题目链接：https://leetcode.cn/problems/linked-list-cycle/

示例 1:
输入: head = [3,2,0,-4], pos = 1
输出: true
解释: 链表中有一个环，其尾部连接到第二个节点。

示例 2:
输入: head = [1,2], pos = 0
输出: true
解释: 链表中有一个环，其尾部连接到第一个节点。

示例 3:
输入: head = [1], pos = -1
输出: false
解释: 链表中没有环。

提示：
- 链表中节点的数目范围是 [0, 10^4]
- -10^5 <= Node.val <= 10^5
- pos 为 -1 或者链表中的一个有效索引
"""
from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1：有环
    # 创建链表: 3->2->0->-4->2 (环)
    head1 = ListNode(3)
    node2 = ListNode(2)
    node3 = ListNode(0)
    node4 = ListNode(-4)
    head1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node2  # 形成环
    assert solution.hasCycle(head1) == True
    
    # 测试用例2：无环
    head2 = ListNode(1)
    head2.next = ListNode(2)
    assert solution.hasCycle(head2) == False
    
    # 测试用例3：单个节点无环
    head3 = ListNode(1)
    assert solution.hasCycle(head3) == False
    
    # 测试用例4：空链表
    assert solution.hasCycle(None) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
