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