# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return None
        
        # Phase 1: Detect if there's a cycle using Floyd's algorithm
        slow = pHead
        fast = pHead
        
        # Move slow one step and fast two steps
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            # If they meet, there's a cycle
            if slow == fast:
                break
        else:
            # No cycle found
            return None
        
        # Phase 2: Find the entrance of the cycle
        # Reset slow to head, keep fast at meeting point
        slow = pHead
        
        # Move both one step at a time until they meet
        # The meeting point is the cycle entrance
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
        