"""
225. 用队列实现栈
请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（push、top、pop 和 empty）。

实现 MyStack 类：

void push(int x) 将元素 x 压入栈顶。
int pop() 移除并返回栈顶元素。
int top() 返回栈顶元素。
boolean empty() 如果栈是空的，返回 true ；否则，返回 false 。

注意：
- 你只能使用队列的基本操作 —— 也就是 push to back、peek/pop from front、size 和 is empty 这些操作。
- 你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。

题目链接：https://leetcode.cn/problems/implement-stack-using-queues/

示例 1:
输入: ["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
输出: [null, null, null, 2, 2, false]

提示：
- 1 <= x <= 9
- 最多调用100 次 push、pop、top 和 empty
- 每次调用 pop 和 top 都保证栈不为空
"""
from typing import List


class MyStack:
    def __init__(self):
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass

    def push(self, x: int) -> None:
        """
        将元素 x 压入栈顶
        """
        # TODO: 在这里实现你的解法
        pass

    def pop(self) -> int:
        """
        移除并返回栈顶元素
        """
        # TODO: 在这里实现你的解法
        pass

    def top(self) -> int:
        """
        返回栈顶元素
        """
        # TODO: 在这里实现你的解法
        pass

    def empty(self) -> bool:
        """
        如果栈是空的，返回 true ；否则，返回 false
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
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
