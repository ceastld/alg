"""
142. 环形链表 II - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    142. 环形链表 II - 标准解法
    """
    
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        标准解法：快慢指针法（Floyd判圈算法）
        
        解题思路：
        1. 使用快慢指针检测环的存在
        2. 如果存在环，找到快慢指针的相遇点
        3. 将一个指针重置到头部，两个指针同时移动
        4. 再次相遇的点就是环的入口
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not head or not head.next:
            return None
        
        # 第一阶段：检测环的存在
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            # 没有环
            return None
        
        # 第二阶段：找到环的入口
        # 将一个指针重置到头部
        slow = head
        # 两个指针同时移动，直到相遇
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def detectCycle_hash(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        哈希表法：使用集合记录访问过的节点
        
        解题思路：
        1. 遍历链表，记录访问过的节点
        2. 如果遇到重复节点，说明有环
        3. 返回第一个重复的节点
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        visited = set()
        current = head
        
        while current:
            if current in visited:
                return current
            visited.add(current)
            current = current.next
        
        return None


def main():
    """测试标准答案"""
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
