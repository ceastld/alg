from typing import List, Optional


class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: Optional[ListNode] = None


#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param lists ListNode类一维数组
# @return ListNode类
#
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # write code here
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])
        return self.Merge(left, right)

    def Merge(self, pHead1: Optional[ListNode], pHead2: Optional[ListNode]) -> Optional[ListNode]:
        p1 = pHead1
        p2 = pHead2
        dummy = ListNode(-1001)
        p = dummy
        while p1 is not None and p2 is not None:
            if p1.val < p2.val:
                p.next = p1
                p1 = p1.next
            else:
                p.next = p2
                p2 = p2.next
            p = p.next
        if p1 is not None:
            p.next = p1
        if p2 is not None:
            p.next = p2
        return dummy.next


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create a linked list from a list of values"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Helper function to convert a linked list to a list of values"""
    result = []
    current = head
    while current is not None:
        result.append(current.val)
        current = current.next
    return result


def test_merge_k_lists():
    """Test cases for mergeKLists function"""
    solution = Solution()
    
    # Test case 1: [{1,2,3},{4,5,6,7}]
    list1 = create_linked_list([1, 2, 3])
    list2 = create_linked_list([4, 5, 6, 7])
    result1 = solution.mergeKLists([list1, list2])
    assert linked_list_to_list(result1) == [1, 2, 3, 4, 5, 6, 7]
    print("Test case 1 passed: [{1,2,3},{4,5,6,7}] -> [1,2,3,4,5,6,7]")
    
    # Test case 2: [{1,2},{1,4,5},{6}]
    list3 = create_linked_list([1, 2])
    list4 = create_linked_list([1, 4, 5])
    list5 = create_linked_list([6])
    result2 = solution.mergeKLists([list3, list4, list5])
    assert linked_list_to_list(result2) == [1, 1, 2, 4, 5, 6]
    print("Test case 2 passed: [{1,2},{1,4,5},{6}] -> [1,1,2,4,5,6]")
    
    # Test case 3: Empty list
    result3 = solution.mergeKLists([])
    assert result3 is None
    print("Test case 3 passed: [] -> None")
    
    # Test case 4: Single list
    list6 = create_linked_list([1, 3, 5])
    result4 = solution.mergeKLists([list6])
    assert linked_list_to_list(result4) == [1, 3, 5]
    print("Test case 4 passed: [{1,3,5}] -> [1,3,5]")
    
    # Test case 5: Lists with None values
    list7 = create_linked_list([1, 2])
    result5 = solution.mergeKLists([list7, None, None])
    assert linked_list_to_list(result5) == [1, 2]
    print("Test case 5 passed: [{1,2}, None, None] -> [1,2]")
    
    print("All test cases passed!")


if __name__ == "__main__":
    test_merge_k_lists()
