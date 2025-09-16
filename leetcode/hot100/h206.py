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