"""
面试题 02.07. 链表相交
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。

图示两个链表在节点 c1 开始相交：

题目数据 保证 整个链式结构中不存在环。

注意，函数返回结果后，链表必须 保持其原始结构 。

题目链接：https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/

示例 1:
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

示例 2:
输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

示例 3:
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。

提示：
- listA 中节点数目为 m
- listB 中节点数目为 n
- 0 <= m, n <= 3 * 104
- 1 <= Node.val <= 105
- 0 <= skipA <= m
- 0 <= skipB <= n
- 如果 listA 和 listB 没有交点，intersectVal 为 0
- 如果 listA 和 listB 有交点，intersectVal == listA[skipA] == listB[skipB]
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    面试题 02.07. 链表相交
    链表操作经典题目
    """
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        p = headA
        q = headB
        while p != q:
            p = p.next if p else headB
            q = q.next if q else headA
        return p


def main():
    """测试用例"""
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
