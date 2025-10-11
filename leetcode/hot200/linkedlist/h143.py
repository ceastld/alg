"""
143. 重排链表
给定一个单链表 L 的头节点 head，单链表 L 表示为：

L0 → L1 → … → Ln - 1 → Ln
请将其重新排列为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

题目链接：https://leetcode.cn/problems/reorder-list/

示例 1:
输入: head = [1,2,3,4]
输出: [1,4,2,3]

示例 2:
输入: head = [1,2,3,4,5]
输出: [1,5,2,4,3]

提示：
- 链表的长度范围为 [1, 5 * 10^4]
- 1 <= node.val <= 1000
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        标准解法：找中点 + 反转 + 合并
        
        解题思路：
        1. 找到链表的中点，将链表分成两部分
        2. 反转后半部分链表
        3. 将前半部分和反转后的后半部分交替合并
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not head or not head.next:
            return
        
        # 步骤1：找到中点
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # 步骤2：分割链表
        second_half = slow.next
        slow.next = None
        
        # 步骤3：反转后半部分
        prev = None
        current = second_half
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        # 步骤4：合并两个链表
        first = head
        second = prev
        while second:
            temp1 = first.next
            temp2 = second.next
            first.next = second
            second.next = temp1
            first = temp1
            second = temp2
        


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表: 1->2->3->4
    head1 = ListNode(1)
    head1.next = ListNode(2)
    head1.next.next = ListNode(3)
    head1.next.next.next = ListNode(4)
    
    solution.reorderList(head1)
    # 验证结果: 1->4->2->3
    assert head1.val == 1
    assert head1.next.val == 4
    assert head1.next.next.val == 2
    assert head1.next.next.next.val == 3
    assert head1.next.next.next.next is None
    
    # 测试用例2
    # 创建链表: 1->2->3->4->5
    head2 = ListNode(1)
    head2.next = ListNode(2)
    head2.next.next = ListNode(3)
    head2.next.next.next = ListNode(4)
    head2.next.next.next.next = ListNode(5)
    
    solution.reorderList(head2)
    # 验证结果: 1->5->2->4->3
    assert head2.val == 1
    assert head2.next.val == 5
    assert head2.next.next.val == 2
    assert head2.next.next.next.val == 4
    assert head2.next.next.next.next.val == 3
    assert head2.next.next.next.next.next is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
