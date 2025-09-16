class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param head ListNode类 
# @param n int整型 
# @return ListNode类
#
class Solution:
    def removeNthFromEnd(self , head: ListNode, n: int) -> ListNode:
        # write code here
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        q = p
        for _ in range(n):
            q = q.next
            if q == None:
                return head
        while q.next != None:
            q = q.next
            p = p.next
        p.next = p.next.next
        return dummy.next


