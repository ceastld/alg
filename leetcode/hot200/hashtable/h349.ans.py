"""
349. 两个数组的交集 - 标准答案
"""
from typing import List


class Solution:
    """
    349. 两个数组的交集 - 标准解法
    """
    
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        标准解法：哈希表法
        
        解题思路：
        1. 将nums1转换为集合，去重
        2. 遍历nums2，检查每个元素是否在nums1的集合中
        3. 如果在，加入结果集合
        4. 返回结果列表
        
        时间复杂度：O(m + n)
        空间复杂度：O(m + n)
        """
        # 将nums1转换为集合，去重
        nums1_set = set(nums1)
        result = set()
        
        # 遍历nums2，找交集
        for num in nums2:
            if num in nums1_set:
                result.add(num)
        
        return list(result)
    
    def intersection_two_sets(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        双集合法
        
        解题思路：
        1. 将两个数组都转换为集合
        2. 使用集合的交集操作
        3. 返回结果列表
        
        时间复杂度：O(m + n)
        空间复杂度：O(m + n)
        """
        return list(set(nums1) & set(nums2))
    
    def intersection_sort(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        排序法
        
        解题思路：
        1. 对两个数组排序
        2. 使用双指针找交集
        3. 去重并返回结果
        
        时间复杂度：O(m log m + n log n)
        空间复杂度：O(1) 不考虑输出数组
        """
        nums1.sort()
        nums2.sort()
        
        result = []
        i = j = 0
        
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                # 避免重复添加
                if not result or result[-1] != nums1[i]:
                    result.append(nums1[i])
                i += 1
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                j += 1
        
        return result
    
    def intersection_binary_search(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        二分查找法
        
        解题思路：
        1. 对nums1排序
        2. 遍历nums2，对每个元素在nums1中二分查找
        3. 找到的元素加入结果集合
        4. 返回结果列表
        
        时间复杂度：O(m log m + n log m)
        空间复杂度：O(min(m, n))
        """
        nums1.sort()
        result = set()
        
        for num in nums2:
            if self.binary_search(nums1, num):
                result.add(num)
        
        return list(result)
    
    def binary_search(self, nums: List[int], target: int) -> bool:
        """二分查找"""
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    result1 = solution.intersection([1, 2, 2, 1], [2, 2])
    assert set(result1) == {2}
    
    # 测试用例2
    result2 = solution.intersection([4, 9, 5], [9, 4, 9, 8, 4])
    assert set(result2) == {9, 4}
    
    # 测试用例3
    result3 = solution.intersection([1, 2, 3], [4, 5, 6])
    assert result3 == []
    
    # 测试用例4
    result4 = solution.intersection([1, 2, 3], [1, 2, 3])
    assert set(result4) == {1, 2, 3}
    
    # 测试用例5
    result5 = solution.intersection([1], [1])
    assert result5 == [1]
    
    # 测试用例6
    result6 = solution.intersection([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    assert set(result6) == {3, 4, 5}
    
    # 测试用例7
    result7 = solution.intersection([], [1, 2, 3])
    assert result7 == []
    
    # 测试用例8
    result8 = solution.intersection([1, 2, 3], [])
    assert result8 == []
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
