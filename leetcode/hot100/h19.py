"""
LeetCode 19. Remove Nth Node From End of List

题目描述：
给你一个链表，删除链表的倒数第n个结点，并且返回链表的头结点。

示例：
head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

数据范围：
- 链表中结点的数目为 sz
- 1 <= sz <= 30
- 0 <= Node.val <= 100
- 1 <= n <= sz
"""

class Solution:
    def removeNthFromEnd(self, head, n: int):
        dummy = ListNode(0)
        dummy.next = head
        
        first = second = dummy
        
        # 让first先走n+1步
        for _ in range(n + 1):
            first = first.next
        
        # 同时移动first和second
        while first:
            first = first.next
            second = second.next
        
        # 删除节点
        second.next = second.next.next
        
        return dummy.next
