"""
203. 移除链表元素
给你一个链表的头节点 head 和一个整数 val ，
请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。

题目链接：https://leetcode.cn/problems/remove-linked-list-elements/

示例 1:
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]

示例 2:
输入：head = [], val = 1
输出：[]

示例 3:
输入：head = [7,7,7,7], val = 7
输出：[]

提示：
- 列表中的节点数目在范围 [0, 104] 内
- 1 <= Node.val <= 50
- 0 <= val <= 50
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    203. 移除链表元素
    链表操作经典题目
    """
    
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        return dummy.next


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表 [1,2,6,3,4,5,6]
    head1 = ListNode(1)
    head1.next = ListNode(2)
    head1.next.next = ListNode(6)
    head1.next.next.next = ListNode(3)
    head1.next.next.next.next = ListNode(4)
    head1.next.next.next.next.next = ListNode(5)
    head1.next.next.next.next.next.next = ListNode(6)
    
    result1 = solution.removeElements(head1, 6)
    # 验证结果应该是 [1,2,3,4,5]
    assert result1.val == 1
    assert result1.next.val == 2
    assert result1.next.next.val == 3
    assert result1.next.next.next.val == 4
    assert result1.next.next.next.next.val == 5
    assert result1.next.next.next.next.next is None
    
    # 测试用例2
    result2 = solution.removeElements(None, 1)
    assert result2 is None
    
    # 测试用例3
    # 创建链表 [7,7,7,7]
    head3 = ListNode(7)
    head3.next = ListNode(7)
    head3.next.next = ListNode(7)
    head3.next.next.next = ListNode(7)
    
    result3 = solution.removeElements(head3, 7)
    assert result3 is None
    
    # 测试用例4
    # 创建链表 [1,2,3]
    head4 = ListNode(1)
    head4.next = ListNode(2)
    head4.next.next = ListNode(3)
    
    result4 = solution.removeElements(head4, 4)
    # 验证结果应该是 [1,2,3]
    assert result4.val == 1
    assert result4.next.val == 2
    assert result4.next.next.val == 3
    assert result4.next.next.next is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
