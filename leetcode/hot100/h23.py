"""
LeetCode 23. Merge k Sorted Lists

题目描述：
给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

示例：
lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]

数据范围：
- k == lists.length
- 0 <= k <= 10^4
- 0 <= lists[i].length <= 500
- -10^4 <= lists[i][j] <= 10^4
- lists[i] 按升序排列
- lists[i].length 的总和不超过 10^4
"""

class Solution:
    def mergeKLists(self, lists: list) -> list:
        if not lists:
            return None
        
        # 方法1：归并排序思路
        def mergeTwoLists(l1, l2):
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 or l2
            return dummy.next
        
        # 归并排序：两两合并
        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged.append(mergeTwoLists(l1, l2))
            lists = merged
        
        return lists[0]
    
    def mergeKListsHeap(self, lists: list) -> list:
        import heapq
        
        # 方法2：堆排序思路
        # 过滤空链表
        lists = [head for head in lists if head]
        if not lists:
            return None
        
        # 创建最小堆
        heap = []
        for i, head in enumerate(lists):
            heapq.heappush(heap, (head.val, i, head))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, i, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        
        return dummy.next
