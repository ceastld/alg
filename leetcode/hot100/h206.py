"""
LeetCode 206. Reverse Linked List

题目描述：
给你单链表的头节点head，请你反转链表，并返回反转后的链表。

示例：
head = [1,2,3,4,5]
输出：[5,4,3,2,1]

数据范围：
- 链表中节点的数目范围是[0, 5000]
- -5000 <= Node.val <= 5000
"""

# Definition for singly-linked list.
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        while current is not None:
            next = current.next
            current.next = prev
            prev = current
            current = next
        return prev

# Test cases
if __name__ == "__main__":
    solution = Solution()
    print(solution.reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))