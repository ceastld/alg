"""
160. 相交链表 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        标准解法：双指针
        
        解题思路：
        1. 使用两个指针分别从两个链表头开始遍历
        2. 当一个指针到达链表末尾时，将其指向另一个链表的头部
        3. 如果两个链表相交，两个指针会在交点相遇
        4. 如果两个链表不相交，两个指针会同时到达None
        
        时间复杂度：O(m + n)
        空间复杂度：O(1)
        """
        if not headA or not headB:
            return None
        
        pA = headA
        pB = headB
        
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        return pA


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1：有相交
    # 创建链表A: 4->1->8->4->5
    headA = ListNode(4)
    headA.next = ListNode(1)
    headA.next.next = ListNode(8)
    headA.next.next.next = ListNode(4)
    headA.next.next.next.next = ListNode(5)
    
    # 创建链表B: 5->6->1->8->4->5 (与A在节点8相交)
    headB = ListNode(5)
    headB.next = ListNode(6)
    headB.next.next = ListNode(1)
    headB.next.next.next = headA.next.next  # 指向A的节点8
    
    result = solution.getIntersectionNode(headA, headB)
    assert result is not None
    assert result.val == 8
    
    # 测试用例2：无相交
    headA2 = ListNode(2)
    headA2.next = ListNode(6)
    headA2.next.next = ListNode(4)
    
    headB2 = ListNode(1)
    headB2.next = ListNode(5)
    
    result2 = solution.getIntersectionNode(headA2, headB2)
    assert result2 is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
