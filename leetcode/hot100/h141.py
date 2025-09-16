"""
LeetCode 141. Linked List Cycle

题目描述：
给你一个链表的头节点head，判断链表中是否有环。
如果链表中存在环，则返回true。否则，返回false。

示例：
head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

数据范围：
- 链表中节点的数目范围是[0, 10^4]
- -10^5 <= Node.val <= 10^5
- pos为-1或者链表中的一个有效索引
"""

# Definition for singly-linked list.
from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False
        slow = head
        fast = head.next
        while fast and fast.next:
            if slow == fast:
                return True
            slow = slow.next
            fast = fast.next.next
        return False