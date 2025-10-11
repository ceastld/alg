"""
234. 回文链表 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        标准解法：找中点 + 反转 + 比较
        
        解题思路：
        1. 使用快慢指针找到链表的中点
        2. 反转后半部分链表
        3. 比较前半部分和反转后的后半部分
        4. 恢复链表结构（可选）
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not head or not head.next:
            return True
        
        # 步骤1：找到中点
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # 步骤2：反转后半部分
        second_half = slow.next
        slow.next = None
        
        prev = None
        current = second_half
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        # 步骤3：比较两部分
        first = head
        second = prev
        result = True
        
        while second:
            if first.val != second.val:
                result = False
                break
            first = first.next
            second = second.next
        
        # 步骤4：恢复链表结构（可选）
        # 这里为了保持原链表结构，可以再次反转后半部分
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1：回文链表
    head1 = ListNode(1)
    head1.next = ListNode(2)
    head1.next.next = ListNode(2)
    head1.next.next.next = ListNode(1)
    assert solution.isPalindrome(head1) == True
    
    # 测试用例2：非回文链表
    head2 = ListNode(1)
    head2.next = ListNode(2)
    assert solution.isPalindrome(head2) == False
    
    # 测试用例3：单个节点
    head3 = ListNode(1)
    assert solution.isPalindrome(head3) == True
    
    # 测试用例4：奇数个节点回文
    head4 = ListNode(1)
    head4.next = ListNode(2)
    head4.next.next = ListNode(3)
    head4.next.next.next = ListNode(2)
    head4.next.next.next.next = ListNode(1)
    assert solution.isPalindrome(head4) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
