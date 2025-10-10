"""
面试题 02.07. 链表相交 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    面试题 02.07. 链表相交 - 标准解法
    """
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        标准解法：双指针法
        
        解题思路：
        1. 使用两个指针分别从两个链表头开始遍历
        2. 当一个指针到达链表末尾时，让它从另一个链表的头开始
        3. 这样两个指针会同时到达相交节点（如果存在）
        4. 如果两个链表不相交，两个指针会同时到达None
        
        时间复杂度：O(m + n)
        空间复杂度：O(1)
        """
        if not headA or not headB:
            return None
        
        pA = headA
        pB = headB
        
        # 两个指针同时移动，直到相遇或都到达None
        while pA != pB:
            # 如果pA到达末尾，从headB开始
            pA = headB if pA is None else pA.next
            # 如果pB到达末尾，从headA开始
            pB = headA if pB is None else pB.next
        
        return pA
    
    def getIntersectionNode_length(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        长度差法：先计算长度差，再同步遍历
        
        解题思路：
        1. 分别计算两个链表的长度
        2. 让长链表的指针先走长度差步
        3. 然后两个指针同时移动，直到相遇
        
        时间复杂度：O(m + n)
        空间复杂度：O(1)
        """
        if not headA or not headB:
            return None
        
        # 计算两个链表的长度
        lenA = self.getLength(headA)
        lenB = self.getLength(headB)
        
        # 让长链表的指针先走长度差步
        pA, pB = headA, headB
        if lenA > lenB:
            for _ in range(lenA - lenB):
                pA = pA.next
        else:
            for _ in range(lenB - lenA):
                pB = pB.next
        
        # 两个指针同时移动，直到相遇
        while pA != pB:
            pA = pA.next
            pB = pB.next
        
        return pA
    
    def getLength(self, head: ListNode) -> int:
        """计算链表长度"""
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        return length


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表A: [4,1,8,4,5]
    headA1 = ListNode(4)
    headA1.next = ListNode(1)
    headA1.next.next = ListNode(8)
    headA1.next.next.next = ListNode(4)
    headA1.next.next.next.next = ListNode(5)
    
    # 创建链表B: [5,0,1,8,4,5] (相交于节点8)
    headB1 = ListNode(5)
    headB1.next = ListNode(0)
    headB1.next.next = ListNode(1)
    headB1.next.next.next = headA1.next.next  # 相交于节点8
    
    result1 = solution.getIntersectionNode(headA1, headB1)
    assert result1 is not None
    assert result1.val == 8
    
    # 测试用例2
    # 创建链表A: [0,9,1,2,4]
    headA2 = ListNode(0)
    headA2.next = ListNode(9)
    headA2.next.next = ListNode(1)
    headA2.next.next.next = ListNode(2)
    headA2.next.next.next.next = ListNode(4)
    
    # 创建链表B: [3,2,4] (相交于节点2)
    headB2 = ListNode(3)
    headB2.next = ListNode(2)
    headB2.next.next = headA2.next.next.next  # 相交于节点2
    
    result2 = solution.getIntersectionNode(headA2, headB2)
    assert result2 is not None
    assert result2.val == 2
    
    # 测试用例3
    # 创建链表A: [2,6,4]
    headA3 = ListNode(2)
    headA3.next = ListNode(6)
    headA3.next.next = ListNode(4)
    
    # 创建链表B: [1,5] (不相交)
    headB3 = ListNode(1)
    headB3.next = ListNode(5)
    
    result3 = solution.getIntersectionNode(headA3, headB3)
    assert result3 is None
    
    # 测试用例4
    # 创建链表A: [1]
    headA4 = ListNode(1)
    
    # 创建链表B: [1] (相交于节点1)
    headB4 = headA4
    
    result4 = solution.getIntersectionNode(headA4, headB4)
    assert result4 is not None
    assert result4.val == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
