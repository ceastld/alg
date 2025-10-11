"""
160. 相交链表
给你两个单链表的头节点 headA 和 headB，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null。

图示两个链表在节点 c1 开始相交：

题目数据保证整个链式结构中不存在环。

注意，函数返回结果后，链表必须保持其原始结构。

题目链接：https://leetcode.cn/problems/intersection-of-two-linked-lists/

示例 1:
输入: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出: Intersected at '8'
解释: 相交节点的值为 8（注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

示例 2:
输入: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出: Intersected at '2'

示例 3:
输入: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出: No intersection

提示：
- listA 中节点数目为 m
- listB 中节点数目为 n
- 1 <= m, n <= 3 * 10^4
- 1 <= Node.val <= 10^5
- 0 <= skipA <= m
- 0 <= skipB <= n
- 如果 listA 和 listB 没有交点，intersectVal 为 0
- 如果 listA 和 listB 有交点，intersectVal == listA[skipA] == listB[skipB]
"""
from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
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
