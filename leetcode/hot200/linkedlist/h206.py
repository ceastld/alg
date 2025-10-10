"""
206. 反转链表
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

题目链接：https://leetcode.cn/problems/reverse-linked-list/

示例 1:
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

示例 2:
输入：head = [1,2]
输出：[2,1]

示例 3:
输入：head = []
输出：[]

提示：
- 链表中节点的数目范围是 [0, 5000]
- -5000 <= Node.val <= 5000
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    206. 反转链表
    链表操作经典题目
    """
    
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        prev = None
        current = head
        while current is not None:
            next = current.next
            current.next = prev
            prev = current
            current = next
        return prev


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表 [1,2,3,4,5]
    head1 = ListNode(1)
    head1.next = ListNode(2)
    head1.next.next = ListNode(3)
    head1.next.next.next = ListNode(4)
    head1.next.next.next.next = ListNode(5)
    
    result1 = solution.reverseList(head1)
    # 验证结果应该是 [5,4,3,2,1]
    assert result1.val == 5
    assert result1.next.val == 4
    assert result1.next.next.val == 3
    assert result1.next.next.next.val == 2
    assert result1.next.next.next.next.val == 1
    assert result1.next.next.next.next.next is None
    
    # 测试用例2
    # 创建链表 [1,2]
    head2 = ListNode(1)
    head2.next = ListNode(2)
    
    result2 = solution.reverseList(head2)
    # 验证结果应该是 [2,1]
    assert result2.val == 2
    assert result2.next.val == 1
    assert result2.next.next is None
    
    # 测试用例3
    result3 = solution.reverseList(None)
    assert result3 is None
    
    # 测试用例4
    # 创建链表 [1]
    head4 = ListNode(1)
    result4 = solution.reverseList(head4)
    assert result4.val == 1
    assert result4.next is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
