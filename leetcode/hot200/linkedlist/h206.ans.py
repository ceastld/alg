"""
206. 反转链表 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    206. 反转链表 - 标准解法
    """
    
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        标准解法：迭代法
        
        解题思路：
        1. 使用三个指针：prev, current, next
        2. 遍历链表，逐个反转节点
        3. 每次将current.next指向prev
        4. 移动指针继续处理下一个节点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        prev = None
        current = head
        
        while current:
            # 保存下一个节点
            next_node = current.next
            # 反转当前节点
            current.next = prev
            # 移动指针
            prev = current
            current = next_node
        
        return prev
    
    def reverseList_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        递归解法：递归法
        
        解题思路：
        1. 递归到链表末尾
        2. 从后往前逐个反转节点
        3. 返回新的头节点
        
        时间复杂度：O(n)
        空间复杂度：O(n) 递归调用栈
        """
        # 递归终止条件
        if not head or not head.next:
            return head
        
        # 递归反转剩余部分
        new_head = self.reverseList_recursive(head.next)
        
        # 反转当前节点
        head.next.next = head
        head.next = None
        
        return new_head


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
