"""
LeetCode 234. Palindrome Linked List

题目描述：
给你一个单链表的头节点head，请你判断该链表是否为回文链表。如果是，返回true；否则，返回false。

示例：
head = [1,2,2,1]
输出：true

数据范围：
- 链表中节点数目在范围[1, 10^5]内
- 0 <= Node.val <= 9
"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: bool
        """
        # 快慢指针，一个速度1，一个速度2
        # 慢指针走一步就反转一下链表连接
        # 快指针走到结尾则慢指针走到中间，这时候从中间往两边遍历，判断是否相等
        # 需要细节处理奇数和偶数个节点的情况
        
        if not head or not head.next:
            return True
        
        # Initialize pointers
        slow = fast = head
        prev = None
        
        # Find middle and reverse first half
        while fast and fast.next:
            fast = fast.next.next  # Move fast pointer 2 steps
            # Reverse the connection as we move slow pointer
            next_temp = slow.next
            slow.next = prev
            prev = slow
            slow = next_temp
        
        # Handle odd and even length cases
        if fast:  # Odd number of nodes, skip the middle node
            slow = slow.next
        
        # Compare first half (reversed) with second half
        while prev and slow:
            if prev.val != slow.val:
                return False
            prev = prev.next
            slow = slow.next
        
        return True
        