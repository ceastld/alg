"""
LeetCode 2. Add Two Numbers

题目描述：
给定两个非空链表，表示两个非负整数。数字以逆序存储，每个节点包含一个数字。
将这两个数相加并以相同形式返回一个链表。

示例：
l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807

数据范围：
- 每个链表中的节点数在范围 [1, 100] 内
- 0 <= Node.val <= 9
- 题目数据保证列表表示的数字不含前导零
"""

class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            # 计算当前位的和
            total = carry
            if l1:
                total += l1.val
                l1 = l1.next
            if l2:
                total += l2.val
                l2 = l2.next
            
            # 处理进位
            carry = total // 10
            current.next = ListNode(total % 10)
            current = current.next
        
        return dummy.next
