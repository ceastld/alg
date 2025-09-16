class Solution:
    def findDuplicate(self, nums: list[int]) -> int:
        # 快慢指针找环
        slow = fast = nums[0]
        
        # 第一阶段：找到相遇点
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # 第二阶段：找到环的入口
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        
        return slow
