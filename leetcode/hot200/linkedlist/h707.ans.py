"""
707. 设计链表 - 标准答案
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MyLinkedList:
    """
    707. 设计链表 - 标准解法
    """
    
    def __init__(self):
        """
        标准解法：使用虚拟头节点
        
        解题思路：
        1. 使用虚拟头节点简化边界情况处理
        2. 维护链表长度便于边界检查
        3. 实现基本的链表操作：增删查改
        
        时间复杂度：各操作均为O(n)
        空间复杂度：O(1) 不考虑存储的节点
        """
        self.dummy = ListNode(0)  # 虚拟头节点
        self.size = 0  # 链表长度
    
    def get(self, index: int) -> int:
        """
        获取链表中第index个节点的值
        """
        if index < 0 or index >= self.size:
            return -1
        
        current = self.dummy.next
        for _ in range(index):
            current = current.next
        return current.val
    
    def addAtHead(self, val: int) -> None:
        """
        在链表头部添加节点
        """
        self.addAtIndex(0, val)
    
    def addAtTail(self, val: int) -> None:
        """
        在链表尾部添加节点
        """
        self.addAtIndex(self.size, val)
    
    def addAtIndex(self, index: int, val: int) -> None:
        """
        在指定位置添加节点
        """
        if index > self.size:
            return
        
        if index < 0:
            index = 0
        
        # 找到要插入位置的前一个节点
        current = self.dummy
        for _ in range(index):
            current = current.next
        
        # 创建新节点并插入
        new_node = ListNode(val)
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def deleteAtIndex(self, index: int) -> None:
        """
        删除指定位置的节点
        """
        if index < 0 or index >= self.size:
            return
        
        # 找到要删除节点的前一个节点
        current = self.dummy
        for _ in range(index):
            current = current.next
        
        # 删除节点
        current.next = current.next.next
        self.size -= 1


def main():
    """测试标准答案"""
    # 测试用例1
    linkedList = MyLinkedList()
    linkedList.addAtHead(1)
    linkedList.addAtTail(3)
    linkedList.addAtIndex(1, 2)  # 链表变为1->2->3
    assert linkedList.get(1) == 2
    linkedList.deleteAtIndex(1)  # 现在链表是1->3
    assert linkedList.get(1) == 3
    
    # 测试用例2
    linkedList2 = MyLinkedList()
    linkedList2.addAtHead(1)
    linkedList2.deleteAtIndex(0)
    assert linkedList2.get(0) == -1
    
    # 测试用例3
    linkedList3 = MyLinkedList()
    linkedList3.addAtHead(1)
    linkedList3.addAtTail(3)
    linkedList3.addAtIndex(1, 2)
    assert linkedList3.get(0) == 1
    assert linkedList3.get(1) == 2
    assert linkedList3.get(2) == 3
    
    # 测试用例4
    linkedList4 = MyLinkedList()
    linkedList4.addAtHead(1)
    linkedList4.addAtTail(3)
    linkedList4.addAtIndex(1, 2)
    linkedList4.deleteAtIndex(0)
    assert linkedList4.get(0) == 2
    assert linkedList4.get(1) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
