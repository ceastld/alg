"""
234. 回文链表
给你一个单链表的头节点 head，请你判断该链表是否为回文链表。如果是，返回 true；否则，返回 false。

题目链接：https://leetcode.cn/problems/palindrome-linked-list/

示例 1:
输入: head = [1,2,2,1]
输出: true

示例 2:
输入: head = [1,2]
输出: false

提示：
- 链表中节点数目在范围[1, 10^5] 内
- 0 <= Node.val <= 9

进阶：你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        优化解法：slow移动过程中反转前半部分
        
        解题思路：
        1. slow指针移动时，同时反转前半部分链表
        2. 到达中点后，直接比较前半部分和后半部分
        3. 避免额外的反转操作
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not head or not head.next:
            return True
        
        slow = fast = head
        prev = None
        
        # slow移动时反转前半部分
        while fast and fast.next:
            fast = fast.next.next
            next_temp = slow.next
            slow.next = prev
            prev = slow
            slow = next_temp
        
        # 处理奇数长度的情况
        if fast:  # 奇数长度，slow需要再移动一步
            slow = slow.next
        
        # 比较前半部分和后半部分
        while prev and slow:
            if prev.val != slow.val:
                return False
            prev = prev.next
            slow = slow.next
        
        return True


def main():
    """测试用例"""
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
