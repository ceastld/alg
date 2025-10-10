"""
24. 两两交换链表中的节点
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

题目链接：https://leetcode.cn/problems/swap-nodes-in-pairs/

示例 1:
输入：head = [1,2,3,4]
输出：[2,1,4,3]

示例 2:
输入：head = []
输出：[]

示例 3:
输入：head = [1]
输出：[1]

提示：
- 链表中节点的数目在范围 [0, 100] 内
- 0 <= Node.val <= 100
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    24. 两两交换链表中的节点
    链表操作经典题目
    """
    
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        while current.next and current.next.next:
            first = current.next
            second = current.next.next
            first.next = second.next
            second.next = first
            current.next = second
            current = first
        return dummy.next


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表 [1,2,3,4]
    head1 = ListNode(1)
    head1.next = ListNode(2)
    head1.next.next = ListNode(3)
    head1.next.next.next = ListNode(4)
    
    result1 = solution.swapPairs(head1)
    # 验证结果应该是 [2,1,4,3]
    assert result1.val == 2
    assert result1.next.val == 1
    assert result1.next.next.val == 4
    assert result1.next.next.next.val == 3
    assert result1.next.next.next.next is None
    
    # 测试用例2
    result2 = solution.swapPairs(None)
    assert result2 is None
    
    # 测试用例3
    # 创建链表 [1]
    head3 = ListNode(1)
    result3 = solution.swapPairs(head3)
    assert result3.val == 1
    assert result3.next is None
    
    # 测试用例4
    # 创建链表 [1,2,3]
    head4 = ListNode(1)
    head4.next = ListNode(2)
    head4.next.next = ListNode(3)
    
    result4 = solution.swapPairs(head4)
    # 验证结果应该是 [2,1,3]
    assert result4.val == 2
    assert result4.next.val == 1
    assert result4.next.next.val == 3
    assert result4.next.next.next is None
    
    # 测试用例5
    # 创建链表 [1,2,3,4,5,6]
    head5 = ListNode(1)
    head5.next = ListNode(2)
    head5.next.next = ListNode(3)
    head5.next.next.next = ListNode(4)
    head5.next.next.next.next = ListNode(5)
    head5.next.next.next.next.next = ListNode(6)
    
    result5 = solution.swapPairs(head5)
    # 验证结果应该是 [2,1,4,3,6,5]
    assert result5.val == 2
    assert result5.next.val == 1
    assert result5.next.next.val == 4
    assert result5.next.next.next.val == 3
    assert result5.next.next.next.next.val == 6
    assert result5.next.next.next.next.next.val == 5
    assert result5.next.next.next.next.next.next is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
