class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param head ListNode类 
# @param m int整型 
# @param n int整型 
# @return ListNode类
#
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        """
        Reverse the linked list from position m to n (1-indexed)
        Time complexity: O(n), Space complexity: O(1)
        """
        if not head or m == n:
            return head
        
        # Create a dummy node to handle edge cases
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # Move to the node before position m
        for _ in range(m - 1):
            prev = prev.next
        
        # Start reversing from position m
        current = prev.next
        next_node = current.next
        
        # Reverse the sublist from m to n
        for _ in range(n - m):
            current.next = next_node.next
            next_node.next = prev.next
            prev.next = next_node
            next_node = current.next
        
        return dummy.next


def create_linked_list(values: list) -> ListNode:
    """Helper function to create a linked list from a list of values"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def linked_list_to_list(head: ListNode) -> list:
    """Helper function to convert linked list to list for easy comparison"""
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def test_reverse_between():
    """Test cases for reverseBetween function"""
    solution = Solution()
    
    # Test case 1: {1,2,3,4,5}, m=2, n=4 -> {1,4,3,2,5}
    head1 = create_linked_list([1, 2, 3, 4, 5])
    result1 = solution.reverseBetween(head1, 2, 4)
    expected1 = [1, 4, 3, 2, 5]
    actual1 = linked_list_to_list(result1)
    print(f"Test 1: {actual1} == {expected1} ? {actual1 == expected1}")
    
    # Test case 2: {5}, m=1, n=1 -> {5}
    head2 = create_linked_list([5])
    result2 = solution.reverseBetween(head2, 1, 1)
    expected2 = [5]
    actual2 = linked_list_to_list(result2)
    print(f"Test 2: {actual2} == {expected2} ? {actual2 == expected2}")
    
    # Test case 3: {1,2,3,4,5}, m=1, n=5 -> {5,4,3,2,1}
    head3 = create_linked_list([1, 2, 3, 4, 5])
    result3 = solution.reverseBetween(head3, 1, 5)
    expected3 = [5, 4, 3, 2, 1]
    actual3 = linked_list_to_list(result3)
    print(f"Test 3: {actual3} == {expected3} ? {actual3 == expected3}")
    
    # Test case 4: {1,2,3,4,5}, m=3, n=3 -> {1,2,3,4,5}
    head4 = create_linked_list([1, 2, 3, 4, 5])
    result4 = solution.reverseBetween(head4, 3, 3)
    expected4 = [1, 2, 3, 4, 5]
    actual4 = linked_list_to_list(result4)
    print(f"Test 4: {actual4} == {expected4} ? {actual4 == expected4}")


if __name__ == "__main__":
    test_reverse_between()
        