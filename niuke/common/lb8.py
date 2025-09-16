class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param pHead ListNode类 
# @param k int整型 
# @return ListNode类
#
class Solution:
    def FindKthToTail(self , pHead: ListNode, k: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = pHead
        p = dummy
        for _ in range(k):
            p = p.next
            if p == None:
                return None
        while p.next != None:
            p = p.next
            dummy = dummy.next
        return dummy.next
