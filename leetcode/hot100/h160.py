"""
LeetCode 160. Intersection of Two Linked Lists

题目描述：
给你两个单链表的头节点headA和headB，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回null。
题目数据保证整个链式结构中不存在环。注意，函数返回结果后，链表必须保持其原始结构。

示例：
listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'

数据范围：
- listA中节点数目为m
- listB中节点数目为n
- 1 <= m, n <= 3 * 10^4
- 1 <= Node.val <= 10^5
- 0 <= skipA <= m
- 0 <= skipB <= n
- 如果listA和listB没有交点，intersectVal为0
- 如果listA和listB有交点，intersectVal == listA[skipA] == listB[skipB]
"""

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA, headB):
        """
        Find the intersection node of two linked lists using two pointers.

        The key insight is that if we traverse both lists and then switch to the other list
        when we reach the end, both pointers will meet at the intersection node (if it exists).
        This works because both pointers will travel the same total distance: m + n.

        Time Complexity: O(m + n)
        Space Complexity: O(1)

        Args:
            headA: Head of the first linked list
            headB: Head of the second linked list

        Returns:
            The intersection node if exists, None otherwise
        """
        if not headA or not headB:
            return None

        # Two pointers starting from each head
        ptrA, ptrB = headA, headB

        # Traverse both lists
        # When ptrA reaches end of listA, redirect it to headB
        # When ptrB reaches end of listB, redirect it to headA
        # If there's an intersection, they will meet at the intersection node
        # If no intersection, they will both become None at the same time

        while ptrA != ptrB:
            # Move ptrA to next node, or to headB if at end of listA
            ptrA = ptrA.next if ptrA else headB
            # Move ptrB to next node, or to headA if at end of listB
            ptrB = ptrB.next if ptrB else headA

        # Return the intersection node (or None if no intersection)
        return ptrA


def create_linked_list(values):
    """Helper function to create a linked list from a list of values."""
    if not values:
        return None

    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def create_intersected_lists(listA, listB, skipA, skipB):
    """
    Create two linked lists that intersect at the specified position.

    Args:
        listA: Values for the first linked list
        listB: Values for the second linked list
        skipA: Number of nodes to skip in listA before intersection
        skipB: Number of nodes to skip in listB before intersection

    Returns:
        Tuple of (headA, headB) - the heads of the two linked lists
    """
    if skipA >= len(listA) or skipB >= len(listB):
        # No intersection case
        return create_linked_list(listA), create_linked_list(listB)

    # Create the common part (intersection)
    common_start = skipA
    common_values = listA[common_start:]
    common_head = create_linked_list(common_values)

    # Create listA up to intersection point
    headA = create_linked_list(listA[:common_start]) if common_start > 0 else None
    if headA:
        current = headA
        while current.next:
            current = current.next
        current.next = common_head
    else:
        headA = common_head

    # Create listB up to intersection point
    headB = create_linked_list(listB[:skipB]) if skipB > 0 else None
    if headB:
        current = headB
        while current.next:
            current = current.next
        current.next = common_head
    else:
        headB = common_head

    return headA, headB


def test_solution():
    """Test the solution with the provided examples."""
    solution = Solution()

    # Test Case 1: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
    print("Test Case 1:")
    headA, headB = create_intersected_lists([4, 1, 8, 4, 5], [5, 6, 1, 8, 4, 5], 2, 3)
    result = solution.getIntersectionNode(headA, headB)
    print(f"Expected: Intersected at '8', Got: {'Intersected at ' + str(result.val) if result else 'No intersection'}")

    # Test Case 2: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
    print("\nTest Case 2:")
    headA, headB = create_intersected_lists([1, 9, 1, 2, 4], [3, 2, 4], 3, 1)
    result = solution.getIntersectionNode(headA, headB)
    print(f"Expected: Intersected at '2', Got: {'Intersected at ' + str(result.val) if result else 'No intersection'}")

    # Test Case 3: No intersection
    print("\nTest Case 3:")
    headA, headB = create_intersected_lists([2, 6, 4], [1, 5], 3, 2)
    result = solution.getIntersectionNode(headA, headB)
    print(f"Expected: No intersection, Got: {'Intersected at ' + str(result.val) if result else 'No intersection'}")

    # Test Case 4: One list is empty
    print("\nTest Case 4:")
    headA, headB = create_linked_list([1, 2, 3]), None
    result = solution.getIntersectionNode(headA, headB)
    print(f"Expected: No intersection, Got: {'Intersected at ' + str(result.val) if result else 'No intersection'}")


if __name__ == "__main__":
    test_solution()
