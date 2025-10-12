"""
232. 用栈实现队列 - 标准答案
"""
from typing import List


class MyQueue:
    def __init__(self):
        """
        使用两个栈实现队列
        
        解题思路：
        1. 使用两个栈：input_stack 和 output_stack
        2. push操作：直接压入input_stack
        3. pop/peek操作：如果output_stack为空，将input_stack的所有元素转移到output_stack
        4. 这样保证了队列的FIFO特性
        """
        self.input_stack = []
        self.output_stack = []

    def push(self, x: int) -> None:
        """
        将元素 x 推到队列的末尾
        时间复杂度：O(1)
        """
        self.input_stack.append(x)

    def pop(self) -> int:
        """
        从队列的开头移除并返回元素
        时间复杂度：摊还O(1)
        """
        self._move_input_to_output()
        return self.output_stack.pop()

    def peek(self) -> int:
        """
        返回队列开头的元素
        时间复杂度：摊还O(1)
        """
        self._move_input_to_output()
        return self.output_stack[-1]

    def empty(self) -> bool:
        """
        如果队列为空，返回 true ；否则，返回 false
        时间复杂度：O(1)
        """
        return not self.input_stack and not self.output_stack
    
    def _move_input_to_output(self):
        """
        将input_stack的所有元素转移到output_stack
        时间复杂度：O(n)，但摊还分析为O(1)
        """
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())


def main():
    """测试标准答案"""
    # 测试用例1
    myQueue = MyQueue()
    myQueue.push(1)
    myQueue.push(2)
    assert myQueue.peek() == 1
    assert myQueue.pop() == 1
    assert myQueue.empty() == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
