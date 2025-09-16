class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param head ListNode类 
# @param k int整型 
# @return ListNode类
#
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        """
        Reverse nodes in k-group
        
        Args:
            head: Head of the linked list
            k: Number of nodes in each group to reverse
            
        Returns:
            Head of the reversed linked list
        """
        if not head or k == 1:
            return head
            
        # Create dummy node to simplify edge cases
        dummy = ListNode(0)
        dummy.next = head
        
        # prev_group_end: end of previous group
        # current_group_start: start of current group
        prev_group_end = dummy
        current_group_start = head
        
        while True:
            # Check if we have at least k nodes remaining
            count = 0
            temp = current_group_start
            while temp and count < k:
                temp = temp.next
                count += 1
            
            # If we have less than k nodes, keep them as is
            if count < k:
                break
            
            # Reverse the current group of k nodes
            prev = None
            current = current_group_start
            
            for _ in range(k):
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            
            # Connect the reversed group to the previous group
            prev_group_end.next = prev
            current_group_start.next = current
            
            # Update pointers for next iteration
            prev_group_end = current_group_start
            current_group_start = current
        
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
    """Helper function to convert linked list to list for easy verification"""
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def test_solution():
    """Test the solution with provided examples"""
    solution = Solution()
    
    # Test case 1: {1,2,3,4,5}, k=2 -> {2,1,4,3,5}
    head1 = create_linked_list([1, 2, 3, 4, 5])
    result1 = solution.reverseKGroup(head1, 2)
    print(f"Test 1: {linked_list_to_list(result1)}")  # Expected: [2, 1, 4, 3, 5]
    
    # Test case 2: {}, k=1 -> {}
    head2 = create_linked_list([])
    result2 = solution.reverseKGroup(head2, 1)
    print(f"Test 2: {linked_list_to_list(result2)}")  # Expected: []
    
    # Test case 3: {1,2,3,4,5}, k=3 -> {3,2,1,4,5}
    head3 = create_linked_list([1, 2, 3, 4, 5])
    result3 = solution.reverseKGroup(head3, 3)
    print(f"Test 3: {linked_list_to_list(result3)}")  # Expected: [3, 2, 1, 4, 5]
    
    # Test case 4: Single node
    head4 = create_linked_list([1])
    result4 = solution.reverseKGroup(head4, 1)
    print(f"Test 4: {linked_list_to_list(result4)}")  # Expected: [1]
    
    # Test case 5: k=1 (no reversal)
    head5 = create_linked_list([1, 2, 3, 4, 5])
    result5 = solution.reverseKGroup(head5, 1)
    print(f"Test 5: {linked_list_to_list(result5)}")  # Expected: [1, 2, 3, 4, 5]


if __name__ == "__main__":
    test_solution()
    