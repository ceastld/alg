"""
LeetCode 142. Linked List Cycle II

题目描述：
给定一个链表的头节点head，返回链表开始入环的第一个节点。如果链表无环，则返回null。
如果链表中存在环，则返回环的起始节点。

示例：
head = [3,2,0,-4], pos = 1
输出：返回索引为1的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

数据范围：
- 链表中节点的数目范围在范围[0, 10^4]内
- -10^5 <= Node.val <= 10^5
- pos的值为-1或者链表中的一个有效索引
"""

# Definition for singly-linked list.
from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None
        
        # 第一阶段：检测环
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            # 如果循环正常结束，说明无环
            return None
        
        # 第二阶段：找到环的起始节点
        # 将一个指针重置到头部，两个指针都每次走1步
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow