"""
19. 删除链表的倒数第N个节点 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    19. 删除链表的倒数第N个节点 - 标准解法
    """
    
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        标准解法：双指针法（快慢指针）
        
        解题思路：
        1. 使用虚拟头节点简化边界情况处理
        2. 快指针先走n+1步，慢指针保持不动
        3. 然后快慢指针同时移动，直到快指针到达末尾
        4. 此时慢指针指向要删除节点的前一个节点
        5. 删除目标节点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 创建虚拟头节点
        dummy = ListNode(0)
        dummy.next = head
        
        # 快慢指针都从虚拟头节点开始
        fast = dummy
        slow = dummy
        
        # 快指针先走n+1步
        for _ in range(n + 1):
            fast = fast.next
        
        # 快慢指针同时移动，直到快指针到达末尾
        while fast:
            fast = fast.next
            slow = slow.next
        
        # 删除目标节点
        slow.next = slow.next.next
        
        return dummy.next
    
    def removeNthFromEnd_two_pass(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        两次遍历法：先计算长度，再删除
        
        解题思路：
        1. 第一次遍历计算链表长度
        2. 第二次遍历找到要删除的节点
        3. 删除目标节点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 创建虚拟头节点
        dummy = ListNode(0)
        dummy.next = head
        
        # 第一次遍历计算链表长度
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # 第二次遍历找到要删除的节点
        current = dummy
        for _ in range(length - n):
            current = current.next
        
        # 删除目标节点
        current.next = current.next.next
        
        return dummy.next


def main():
    """测试标准答案"""
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
