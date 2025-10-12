"""
225. 用队列实现栈 - 标准答案
"""
from typing import List
from collections import deque


class MyStack:
    def __init__(self):
        """
        使用两个队列实现栈
        
        解题思路：
        1. 使用两个队列：queue1 和 queue2
        2. push操作：将元素加入queue1
        3. pop/top操作：将queue1中除最后一个元素外的所有元素转移到queue2，然后操作最后一个元素
        4. 这样保证了栈的LIFO特性
        """
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x: int) -> None:
        """
        将元素 x 压入栈顶
        时间复杂度：O(1)
        """
        self.queue1.append(x)

    def pop(self) -> int:
        """
        移除并返回栈顶元素
        时间复杂度：O(n)
        """
        while len(self.queue1) > 1:
            self.queue2.append(self.queue1.popleft())
        
        result = self.queue1.popleft()
        self.queue1, self.queue2 = self.queue2, self.queue1
        return result

    def top(self) -> int:
        """
        返回栈顶元素
        时间复杂度：O(n)
        """
        while len(self.queue1) > 1:
            self.queue2.append(self.queue1.popleft())
        
        result = self.queue1[0]
        self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1
        return result

    def empty(self) -> bool:
        """
        如果栈是空的，返回 true ；否则，返回 false
        时间复杂度：O(1)
        """
        return not self.queue1


def main():
    """测试标准答案"""
    # 测试用例1
    myStack = MyStack()
    myStack.push(1)
    myStack.push(2)
    assert myStack.top() == 2
    assert myStack.pop() == 2
    assert myStack.empty() == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
