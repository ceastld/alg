"""
141. 环形链表 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        标准解法：快慢指针（Floyd判圈算法）
        
        解题思路：
        1. 使用两个指针，一个慢指针每次移动一步，一个快指针每次移动两步
        2. 如果链表中有环，快指针最终会追上慢指针
        3. 如果链表中没有环，快指针会先到达链表末尾
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next
        
        while fast and fast.next:
            if slow == fast:
                return True
            slow = slow.next
            fast = fast.next.next
        
        return False


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1：有环
    # 创建链表: 3->2->0->-4->2 (环)
    head1 = ListNode(3)
    node2 = ListNode(2)
    node3 = ListNode(0)
    node4 = ListNode(-4)
    head1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node2  # 形成环
    assert solution.hasCycle(head1) == True
    
    # 测试用例2：无环
    head2 = ListNode(1)
    head2.next = ListNode(2)
    assert solution.hasCycle(head2) == False
    
    # 测试用例3：单个节点无环
    head3 = ListNode(1)
    assert solution.hasCycle(head3) == False
    
    # 测试用例4：空链表
    assert solution.hasCycle(None) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
