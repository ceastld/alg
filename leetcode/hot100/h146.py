"""
LeetCode 146. LRU Cache

题目描述：
请你设计并实现一个满足LRU（最近最少使用）缓存约束的数据结构。
实现LRUCache类：
- LRUCache(int capacity)以正整数作为容量capacity初始化LRU缓存
- int get(int key)如果关键字key存在于缓存中，则返回关键字的值，否则返回-1
- void put(int key, int value)如果关键字key已经存在，则变更其数据值value；如果不存在，则向缓存中插入该组key-value

示例：
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是{1=1}
lRUCache.put(2, 2); // 缓存是{1=1, 2=2}
lRUCache.get(1);    // 返回1
lRUCache.put(3, 3); // 该操作会使得关键字2作废，缓存是{1=1, 3=3}

数据范围：
- 1 <= capacity <= 3000
- 0 <= key <= 10^4
- 0 <= value <= 10^5
- 最多调用2*10^5次get和put
"""

from typing import Optional, Dict

class ListNode:
    """双向链表节点"""
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev: Optional[ListNode] = None
        self.next: Optional[ListNode] = None

class LRUCache:
    """
    LRU Cache - 最近最少使用缓存
    
    实现 LRUCache 类：
    - LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
    - int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1
    - void put(int key, int value) 如果关键字已经存在，则变更其数据值；
      如果关键字不存在，则插入该组「关键字-值」。
      当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间
    """
    
    def __init__(self, capacity: int):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
        """
        self.capacity = capacity
        self.cache: Dict[int, ListNode] = {}  # 哈希表：key -> node
        
        # 创建虚拟头尾节点，简化边界处理
        self.head = ListNode()  # 最近使用的节点
        self.tail = ListNode()  # 最久未使用的节点
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key: int) -> int:
        """
        获取缓存中的值
        
        Args:
            key: 键
            
        Returns:
            值，如果不存在返回-1
        """
        if key in self.cache:
            # 将节点移动到头部（标记为最近使用）
            node = self.cache[key]
            self._move_to_head(node)
            return node.val
        return -1
    
    def put(self, key: int, value: int) -> None:
        """
        设置缓存中的值
        
        Args:
            key: 键
            value: 值
        """
        if key in self.cache:
            # 更新已存在的节点
            node = self.cache[key]
            node.val = value
            self._move_to_head(node)
        else:
            # 创建新节点
            new_node = ListNode(key, value)
            
            if len(self.cache) >= self.capacity:
                # 缓存已满，删除尾部节点（最久未使用）
                tail_node = self._remove_tail()
                del self.cache[tail_node.key]
            
            # 添加新节点到头部
            self.cache[key] = new_node
            self._add_to_head(new_node)
    
    def _add_to_head(self, node: ListNode) -> None:
        """将节点添加到头部"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: ListNode) -> None:
        """删除指定节点"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node: ListNode) -> None:
        """将节点移动到头部"""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _remove_tail(self) -> ListNode:
        """删除尾部节点并返回"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node


class LRUCacheOrderedDict:
    """
    使用OrderedDict实现的LRU Cache - 简化版本
    """
    
    def __init__(self, capacity: int):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
        """
        from collections import OrderedDict
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        """
        获取缓存中的值
        
        Args:
            key: 键
            
        Returns:
            值，如果不存在返回-1
        """
        if key in self.cache:
            # 移动到末尾（标记为最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1
    
    def put(self, key: int, value: int) -> None:
        """
        设置缓存中的值
        
        Args:
            key: 键
            value: 值
        """
        if key in self.cache:
            # 更新已存在的值
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # 删除最久未使用的项（第一个）
                self.cache.popitem(last=False)
            
            # 添加新项
            self.cache[key] = value


class LRUCacheSimple:
    """
    简化版本 - 使用列表和字典
    """
    
    def __init__(self, capacity: int):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
        """
        self.capacity = capacity
        self.cache: Dict[int, int] = {}
        self.order: list = []  # 记录访问顺序
    
    def get(self, key: int) -> int:
        """
        获取缓存中的值
        
        Args:
            key: 键
            
        Returns:
            值，如果不存在返回-1
        """
        if key in self.cache:
            # 移动到末尾
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
    
    def put(self, key: int, value: int) -> None:
        """
        设置缓存中的值
        
        Args:
            key: 键
            value: 值
        """
        if key in self.cache:
            # 更新已存在的值
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            if len(self.cache) >= self.capacity:
                # 删除最久未使用的项
                oldest = self.order.pop(0)
                del self.cache[oldest]
            
            # 添加新项
            self.cache[key] = value
            self.order.append(key)


# 测试用例
def test_lru_cache():
    """测试LRU Cache功能"""
    
    print("=== 测试双向链表版本 ===")
    lru = LRUCache(2)
    
    # 测试用例1
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"get(1): {lru.get(1)}")  # 1
    
    lru.put(3, 3)  # 删除key=2
    print(f"get(2): {lru.get(2)}")  # -1
    
    lru.put(4, 4)  # 删除key=1
    print(f"get(1): {lru.get(1)}")  # -1
    print(f"get(3): {lru.get(3)}")  # 3
    print(f"get(4): {lru.get(4)}")  # 4
    
    print("\n=== 测试OrderedDict版本 ===")
    lru_od = LRUCacheOrderedDict(2)
    
    lru_od.put(1, 1)
    lru_od.put(2, 2)
    print(f"get(1): {lru_od.get(1)}")  # 1
    
    lru_od.put(3, 3)  # 删除key=2
    print(f"get(2): {lru_od.get(2)}")  # -1
    
    lru_od.put(4, 4)  # 删除key=1
    print(f"get(1): {lru_od.get(1)}")  # -1
    print(f"get(3): {lru_od.get(3)}")  # 3
    print(f"get(4): {lru_od.get(4)}")  # 4
    
    print("\n=== 测试简化版本 ===")
    lru_simple = LRUCacheSimple(2)
    
    lru_simple.put(1, 1)
    lru_simple.put(2, 2)
    print(f"get(1): {lru_simple.get(1)}")  # 1
    
    lru_simple.put(3, 3)  # 删除key=2
    print(f"get(2): {lru_simple.get(2)}")  # -1
    
    lru_simple.put(4, 4)  # 删除key=1
    print(f"get(1): {lru_simple.get(1)}")  # -1
    print(f"get(3): {lru_simple.get(3)}")  # 3
    print(f"get(4): {lru_simple.get(4)}")  # 4
    
    print("\n=== 性能对比测试 ===")
    import time
    
    # 测试大数据量
    test_size = 1000
    capacity = 100
    
    # 双向链表版本
    start_time = time.time()
    lru_perf = LRUCache(capacity)
    for i in range(test_size):
        lru_perf.put(i, i)
        if i % 2 == 0:
            lru_perf.get(i // 2)
    end_time = time.time()
    print(f"双向链表版本耗时: {end_time - start_time:.4f}秒")
    
    # OrderedDict版本
    start_time = time.time()
    lru_od_perf = LRUCacheOrderedDict(capacity)
    for i in range(test_size):
        lru_od_perf.put(i, i)
        if i % 2 == 0:
            lru_od_perf.get(i // 2)
    end_time = time.time()
    print(f"OrderedDict版本耗时: {end_time - start_time:.4f}秒")


if __name__ == "__main__":
    test_lru_cache()
