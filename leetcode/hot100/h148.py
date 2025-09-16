"""
LeetCode 148. Sort List

题目描述：
给你链表的头结点head，请将其按升序排列并返回排序后的链表。

示例：
head = [4,2,1,3]
输出：[1,2,3,4]

数据范围：
- 链表中节点的数目在范围[0, 5 * 10^4]内
- -10^5 <= Node.val <= 10^5
"""

# Definition for singly-linked list.
from typing import Optional, List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        return self.merge(self.sortList(head), self.sortList(mid))
    
    def merge(self, left: Optional[ListNode], right: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy
        while left and right:
            if left.val < right.val:
                current.next = left
                left = left.next
            else:
                current.next = right
                right = right.next
            current = current.next
        if left:
            current.next = left
        if right:
            current.next = right
        return dummy.next