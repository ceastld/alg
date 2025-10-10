"""
203. 移除链表元素 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    203. 移除链表元素 - 标准解法
    """
    
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        标准解法：虚拟头节点法
        
        解题思路：
        1. 使用虚拟头节点简化边界情况处理
        2. 遍历链表，删除所有值等于val的节点
        3. 维护当前节点和下一个节点的关系
        4. 返回虚拟头节点的下一个节点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 创建虚拟头节点
        dummy = ListNode(0)
        dummy.next = head
        
        current = dummy
        
        while current.next:
            if current.next.val == val:
                # 删除当前节点的下一个节点
                current.next = current.next.next
            else:
                # 移动到下一个节点
                current = current.next
        
        return dummy.next


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    # 创建链表 [1,2,6,3,4,5,6]
    head1 = ListNode(1)
    head1.next = ListNode(2)
    head1.next.next = ListNode(6)
    head1.next.next.next = ListNode(3)
    head1.next.next.next.next = ListNode(4)
    head1.next.next.next.next.next = ListNode(5)
    head1.next.next.next.next.next.next = ListNode(6)
    
    result1 = solution.removeElements(head1, 6)
    # 验证结果应该是 [1,2,3,4,5]
    assert result1.val == 1
    assert result1.next.val == 2
    assert result1.next.next.val == 3
    assert result1.next.next.next.next.val == 4
    assert result1.next.next.next.next.val == 5
    assert result1.next.next.next.next.next is None
    
    # 测试用例2
    result2 = solution.removeElements(None, 1)
    assert result2 is None
    
    # 测试用例3
    # 创建链表 [7,7,7,7]
    head3 = ListNode(7)
    head3.next = ListNode(7)
    head3.next.next = ListNode(7)
    head3.next.next.next = ListNode(7)
    
    result3 = solution.removeElements(head3, 7)
    assert result3 is None
    
    # 测试用例4
    # 创建链表 [1,2,3]
    head4 = ListNode(1)
    head4.next = ListNode(2)
    head4.next.next = ListNode(3)
    
    result4 = solution.removeElements(head4, 4)
    # 验证结果应该是 [1,2,3]
    assert result4.val == 1
    assert result4.next.val == 2
    assert result4.next.next.val == 3
    assert result4.next.next.next is None
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
