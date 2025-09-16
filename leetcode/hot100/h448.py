class Solution:
    def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
        # 使用数组本身作为哈希表
        for num in nums:
            index = abs(num) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
        
        # 收集所有正数位置的索引+1
        result = []
        for i in range(len(nums)):
            if nums[i] > 0:
                result.append(i + 1)
        
        return result
