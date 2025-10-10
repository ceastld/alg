"""
142. 环形链表 II
给定一个链表的头节点 head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

不允许修改 链表。

题目链接：https://leetcode.cn/problems/linked-list-cycle-ii/

示例 1:
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2:
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3:
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。

提示：
- 链表中节点的数目范围在范围 [0, 104] 内
- -105 <= Node.val <= 105
- pos 的值为 -1 或者链表中的一个有效索引
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    142. 环形链表 II
    链表操作经典题目
    """
    
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        if not head or not head.next:
            return None
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表 [3,2,0,-4] 带环，pos=1
    head1 = ListNode(3)
    head1.next = ListNode(2)
    head1.next.next = ListNode(0)
    head1.next.next.next = ListNode(-4)
    head1.next.next.next.next = head1.next  # 形成环，连接到索引1的节点
    
    result1 = solution.detectCycle(head1)
    assert result1 is not None
    assert result1.val == 2  # 环的入口节点
    
    # 测试用例2
    # 创建链表 [1,2] 带环，pos=0
    head2 = ListNode(1)
    head2.next = ListNode(2)
    head2.next.next = head2  # 形成环，连接到索引0的节点
    
    result2 = solution.detectCycle(head2)
    assert result2 is not None
    assert result2.val == 1  # 环的入口节点
    
    # 测试用例3
    # 创建链表 [1] 无环
    head3 = ListNode(1)
    result3 = solution.detectCycle(head3)
    assert result3 is None
    
    # 测试用例4
    # 创建链表 [1,2,3,4,5] 无环
    head4 = ListNode(1)
    head4.next = ListNode(2)
    head4.next.next = ListNode(3)
    head4.next.next.next = ListNode(4)
    head4.next.next.next.next = ListNode(5)
    
    result4 = solution.detectCycle(head4)
    assert result4 is None
    
    # 测试用例5
    # 创建链表 [1,2,3,4,5,6] 带环，pos=2
    head5 = ListNode(1)
    head5.next = ListNode(2)
    head5.next.next = ListNode(3)
    head5.next.next.next = ListNode(4)
    head5.next.next.next.next = ListNode(5)
    head5.next.next.next.next.next = ListNode(6)
    head5.next.next.next.next.next.next = head5.next.next  # 形成环，连接到索引2的节点
    
    result5 = solution.detectCycle(head5)
    assert result5 is not None
    assert result5.val == 3  # 环的入口节点
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
