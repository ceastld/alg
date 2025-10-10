"""
24. 两两交换链表中的节点 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    24. 两两交换链表中的节点 - 标准解法
    """
    
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        标准解法：虚拟头节点法
        
        解题思路：
        1. 使用虚拟头节点简化边界情况处理
        2. 维护三个指针：prev, first, second
        3. 每次交换两个相邻节点
        4. 更新指针继续处理下一对节点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 创建虚拟头节点
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # 确保有至少两个节点可以交换
        while prev.next and prev.next.next:
            # 获取要交换的两个节点
            first = prev.next
            second = prev.next.next
            
            # 执行交换
            prev.next = second
            first.next = second.next
            second.next = first
            
            # 移动指针到下一对节点
            prev = first
        
        return dummy.next
    
    def swapPairs_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        递归解法：递归法
        
        解题思路：
        1. 递归处理剩余部分
        2. 交换当前两个节点
        3. 返回新的头节点
        
        时间复杂度：O(n)
        空间复杂度：O(n) 递归调用栈
        """
        # 递归终止条件
        if not head or not head.next:
            return head
        
        # 获取要交换的两个节点
        first = head
        second = head.next
        
        # 递归处理剩余部分
        first.next = self.swapPairs_recursive(second.next)
        
        # 交换当前两个节点
        second.next = first
        
        return second


def main():
    """测试标准答案"""
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
