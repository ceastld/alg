"""
707. 设计链表
设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

- get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
- addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
- addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
- addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val 的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
- deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。

题目链接：https://leetcode.cn/problems/design-linked-list/

示例 1:
MyLinkedList linkedList = new MyLinkedList();
linkedList.addAtHead(1);
linkedList.addAtTail(3);
linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
linkedList.get(1);            //返回2
linkedList.deleteAtIndex(1);  //现在链表是1-> 3
linkedList.get(1);            //返回3

提示：
- 0 <= index, val <= 1000
- 请不要使用内置的 LinkedList 库。
- get, addAtHead, addAtTail, addAtIndex 和 deleteAtIndex 的操作次数不超过 2000。
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MyLinkedList:
    """
    707. 设计链表
    链表设计经典题目
    """
    
    def __init__(self):
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        self.dummy = ListNode(0)
        self.size = 0
    
    def get(self, index: int) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        if index < 0 or index >= self.size:
            return -1
        current = self.dummy.next
        for _ in range(index):
            current = current.next
        return current.val
    
    def addAtHead(self, val: int) -> None:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        self.addAtIndex(0, val)
    
    def addAtTail(self, val: int) -> None:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        self.addAtIndex(self.size, val)
    
    def addAtIndex(self, index: int, val: int) -> None:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        if index > self.size:
            return
        if index < 0:
            index = 0
        current = self.dummy
        for _ in range(index):
            current = current.next
        new_node = ListNode(val)
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def deleteAtIndex(self, index: int) -> None:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        if index < 0 or index >= self.size:
            return
        current = self.dummy
        for _ in range(index):
            current = current.next
        current.next = current.next.next
        self.size -= 1


def main():
    """测试用例"""
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
