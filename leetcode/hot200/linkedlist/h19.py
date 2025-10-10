"""
19. 删除链表的倒数第N个节点
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

题目链接：https://leetcode.cn/problems/remove-nth-node-from-end-of-list/

示例 1:
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

示例 2:
输入：head = [1], n = 1
输出：[]

示例 3:
输入：head = [1,2], n = 1
输出：[1]

提示：
- 链表中结点的数目为 sz
- 1 <= sz <= 30
- 0 <= n <= sz
- 1 <= Node.val <= 100
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    19. 删除链表的倒数第N个节点
    链表操作经典题目
    """
    
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        dummy = ListNode(0)
        dummy.next = head
        slow = dummy
        fast = dummy
        for _ in range(n):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next


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
    
    result1 = solution.removeNthFromEnd(head1, 2)
    # 验证结果应该是 [1,2,3,5]
    assert result1.val == 1
    assert result1.next.val == 2
    assert result1.next.next.val == 3
    assert result1.next.next.next.val == 5
    assert result1.next.next.next.next is None
    
    # 测试用例2
    # 创建链表 [1]
    head2 = ListNode(1)
    result2 = solution.removeNthFromEnd(head2, 1)
    assert result2 is None
    
    # 测试用例3
    # 创建链表 [1,2]
    head3 = ListNode(1)
    head3.next = ListNode(2)
    result3 = solution.removeNthFromEnd(head3, 1)
    # 验证结果应该是 [1]
    assert result3.val == 1
    assert result3.next is None
    
    # 测试用例4
    # 创建链表 [1,2,3,4,5]
    head4 = ListNode(1)
    head4.next = ListNode(2)
    head4.next.next = ListNode(3)
    head4.next.next.next = ListNode(4)
    head4.next.next.next.next = ListNode(5)
    
    result4 = solution.removeNthFromEnd(head4, 5)
    # 验证结果应该是 [2,3,4,5]
    assert result4.val == 2
    assert result4.next.val == 3
    assert result4.next.next.val == 4
    assert result4.next.next.next.val == 5
    assert result4.next.next.next.next is None
    
    # 测试用例5
    # 创建链表 [1,2,3,4,5]
    head5 = ListNode(1)
    head5.next = ListNode(2)
    head5.next.next = ListNode(3)
    head5.next.next.next = ListNode(4)
    head5.next.next.next.next = ListNode(5)
    
    result5 = solution.removeNthFromEnd(head5, 1)
    # 验证结果应该是 [1,2,3,4]
    assert result5.val == 1
    assert result5.next.val == 2
    assert result5.next.next.val == 3
    assert result5.next.next.next.val == 4
    assert result5.next.next.next.next is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
