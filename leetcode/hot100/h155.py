from typing import List, Optional

class MinStack:
    """
    设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈
    
    实现 MinStack 类:
    - MinStack() 初始化堆栈对象
    - void push(int val) 将元素val推入堆栈
    - void pop() 删除堆栈顶部的元素
    - int top() 获取堆栈顶部的元素
    - int getMin() 获取堆栈中的最小元素
    """
    
    def __init__(self):
        """初始化堆栈对象"""
        self.stack: List[int] = []
        self.min_stack: List[int] = []  # 辅助栈，存储最小值
    
    def push(self, val: int) -> None:
        """
        将元素val推入堆栈
        
        Args:
            val: 要推入的元素
        """
        self.stack.append(val)
        
        # 如果min_stack为空，或者val小于等于当前最小值，则推入min_stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self) -> None:
        """删除堆栈顶部的元素"""
        if not self.stack:
            return
        
        val = self.stack.pop()
        
        # 如果弹出的元素是当前最小值，则也从min_stack中弹出
        if self.min_stack and val == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self) -> int:
        """
        获取堆栈顶部的元素
        
        Returns:
            栈顶元素
        """
        return self.stack[-1]
    
    def getMin(self) -> int:
        """
        获取堆栈中的最小元素
        
        Returns:
            最小元素
        """
        return self.min_stack[-1]


class MinStackOptimized:
    """
    优化版本：使用单个栈存储差值，节省空间
    """
    
    def __init__(self):
        """初始化堆栈对象"""
        self.stack: List[int] = []
        self.min_val: Optional[int] = None
    
    def push(self, val: int) -> None:
        """
        将元素val推入堆栈
        
        Args:
            val: 要推入的元素
        """
        if self.min_val is None:
            # 第一个元素
            self.min_val = val
            self.stack.append(0)
        else:
            # 存储与当前最小值的差值
            diff = val - self.min_val
            self.stack.append(diff)
            
            # 如果新元素更小，更新最小值
            if diff < 0:
                self.min_val = val
    
    def pop(self) -> None:
        """删除堆栈顶部的元素"""
        if not self.stack:
            return
        
        diff = self.stack.pop()
        
        if diff < 0:
            # 如果差值为负，说明弹出的元素是最小值
            # 需要恢复之前的最小值
            self.min_val = self.min_val - diff
        
        # 如果栈为空，重置最小值
        if not self.stack:
            self.min_val = None
    
    def top(self) -> int:
        """
        获取堆栈顶部的元素
        
        Returns:
            栈顶元素
        """
        if not self.stack:
            return 0
        
        diff = self.stack[-1]
        
        if diff < 0:
            # 如果差值为负，说明栈顶元素就是最小值
            return self.min_val
        else:
            # 否则，栈顶元素 = 最小值 + 差值
            return self.min_val + diff
    
    def getMin(self) -> int:
        """
        获取堆栈中的最小元素
        
        Returns:
            最小元素
        """
        return self.min_val if self.min_val is not None else 0


class MinStackSimple:
    """
    最简单版本：每次push时重新计算最小值
    时间复杂度：push O(1), pop O(1), top O(1), getMin O(n)
    """
    
    def __init__(self):
        """初始化堆栈对象"""
        self.stack: List[int] = []
    
    def push(self, val: int) -> None:
        """将元素val推入堆栈"""
        self.stack.append(val)
    
    def pop(self) -> None:
        """删除堆栈顶部的元素"""
        if self.stack:
            self.stack.pop()
    
    def top(self) -> int:
        """获取堆栈顶部的元素"""
        return self.stack[-1]
    
    def getMin(self) -> int:
        """获取堆栈中的最小元素"""
        return min(self.stack) if self.stack else 0


# 测试用例
def test_min_stack():
    """测试MinStack功能"""
    print("=== 测试MinStack ===")
    
    # 测试基本版本
    min_stack = MinStack()
    min_stack.push(-2)
    min_stack.push(0)
    min_stack.push(-3)
    print(f"getMin(): {min_stack.getMin()}")  # -3
    min_stack.pop()
    print(f"top(): {min_stack.top()}")        # 0
    print(f"getMin(): {min_stack.getMin()}")  # -2
    
    print("\n=== 测试优化版本 ===")
    
    # 测试优化版本
    min_stack_opt = MinStackOptimized()
    min_stack_opt.push(-2)
    min_stack_opt.push(0)
    min_stack_opt.push(-3)
    print(f"getMin(): {min_stack_opt.getMin()}")  # -3
    min_stack_opt.pop()
    print(f"top(): {min_stack_opt.top()}")        # 0
    print(f"getMin(): {min_stack_opt.getMin()}")  # -2
    
    print("\n=== 测试简单版本 ===")
    
    # 测试简单版本
    min_stack_simple = MinStackSimple()
    min_stack_simple.push(-2)
    min_stack_simple.push(0)
    min_stack_simple.push(-3)
    print(f"getMin(): {min_stack_simple.getMin()}")  # -3
    min_stack_simple.pop()
    print(f"top(): {min_stack_simple.top()}")        # 0
    print(f"getMin(): {min_stack_simple.getMin()}")  # -2


if __name__ == "__main__":
    test_min_stack()
