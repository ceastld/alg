from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        只出现一次的数字 - 异或运算解法
        
        Args:
            nums: 整数数组，除了某个元素只出现一次外，其余每个元素均出现两次
            
        Returns:
            只出现一次的数字
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    def singleNumberHashSet(self, nums: List[int]) -> int:
        """
        只出现一次的数字 - 哈希集合解法
        
        Args:
            nums: 整数数组
            
        Returns:
            只出现一次的数字
        """
        seen = set()
        for num in nums:
            if num in seen:
                seen.remove(num)
            else:
                seen.add(num)
        return seen.pop()
    
    def singleNumberMath(self, nums: List[int]) -> int:
        """
        只出现一次的数字 - 数学解法
        
        Args:
            nums: 整数数组
            
        Returns:
            只出现一次的数字
        """
        # 2 * (a + b + c) - (a + a + b + b + c) = c
        return 2 * sum(set(nums)) - sum(nums)
    
    def singleNumberCounter(self, nums: List[int]) -> int:
        """
        只出现一次的数字 - 计数器解法
        
        Args:
            nums: 整数数组
            
        Returns:
            只出现一次的数字
        """
        from collections import Counter
        count = Counter(nums)
        for num, freq in count.items():
            if freq == 1:
                return num
    
    def singleNumberSort(self, nums: List[int]) -> int:
        """
        只出现一次的数字 - 排序解法
        
        Args:
            nums: 整数数组
            
        Returns:
            只出现一次的数字
        """
        nums.sort()
        for i in range(0, len(nums), 2):
            if i == len(nums) - 1 or nums[i] != nums[i + 1]:
                return nums[i]


# 测试用例
def test_single_number():
    """测试只出现一次的数字功能"""
    solution = Solution()
    
    # 测试用例1
    nums1 = [2, 2, 1]
    result1 = solution.singleNumber(nums1)
    print(f"测试1 - nums: {nums1}")
    print(f"结果: {result1}")  # 1
    print()
    
    # 测试用例2
    nums2 = [4, 1, 2, 1, 2]
    result2 = solution.singleNumber(nums2)
    print(f"测试2 - nums: {nums2}")
    print(f"结果: {result2}")  # 4
    print()
    
    # 测试用例3
    nums3 = [1]
    result3 = solution.singleNumber(nums3)
    print(f"测试3 - nums: {nums3}")
    print(f"结果: {result3}")  # 1
    print()
    
    # 测试用例4
    nums4 = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    result4 = solution.singleNumber(nums4)
    print(f"测试4 - nums: {nums4}")
    print(f"结果: {result4}")  # 5
    print()
    
    # 测试用例5
    nums5 = [-1, -1, -2]
    result5 = solution.singleNumber(nums5)
    print(f"测试5 - nums: {nums5}")
    print(f"结果: {result5}")  # -2
    print()
    
    # 对比不同算法
    print("=== 算法对比 ===")
    test_nums = [4, 1, 2, 1, 2]
    
    xor_result = solution.singleNumber(test_nums)
    hash_result = solution.singleNumberHashSet(test_nums)
    math_result = solution.singleNumberMath(test_nums)
    counter_result = solution.singleNumberCounter(test_nums)
    sort_result = solution.singleNumberSort(test_nums)
    
    print(f"异或运算结果: {xor_result}")
    print(f"哈希集合结果: {hash_result}")
    print(f"数学解法结果: {math_result}")
    print(f"计数器结果: {counter_result}")
    print(f"排序解法结果: {sort_result}")
    print(f"结果一致: {xor_result == hash_result == math_result == counter_result == sort_result}")
    
    # 性能测试
    print("\n=== 性能测试 ===")
    import time
    import random
    
    # 生成测试数据
    large_nums = []
    for i in range(1000):
        large_nums.extend([i, i])  # 每个数字出现两次
    large_nums.append(9999)  # 添加一个只出现一次的数字
    random.shuffle(large_nums)
    
    # 测试异或运算
    start_time = time.time()
    xor_result = solution.singleNumber(large_nums)
    xor_time = time.time() - start_time
    print(f"异或运算耗时: {xor_time:.6f}秒, 结果: {xor_result}")
    
    # 测试哈希集合
    start_time = time.time()
    hash_result = solution.singleNumberHashSet(large_nums)
    hash_time = time.time() - start_time
    print(f"哈希集合耗时: {hash_time:.6f}秒, 结果: {hash_result}")
    
    # 测试数学解法
    start_time = time.time()
    math_result = solution.singleNumberMath(large_nums)
    math_time = time.time() - start_time
    print(f"数学解法耗时: {math_time:.6f}秒, 结果: {math_result}")


# 异或运算原理演示
def demonstrate_xor():
    """演示异或运算的原理"""
    print("=== 异或运算原理演示 ===")
    
    # 异或运算的基本性质
    print("异或运算的基本性质:")
    print(f"a ^ a = 0: {5 ^ 5}")  # 0
    print(f"a ^ 0 = a: {5 ^ 0}")  # 5
    print(f"a ^ b = b ^ a: {5 ^ 3} = {3 ^ 5}")  # 交换律
    print(f"(a ^ b) ^ c = a ^ (b ^ c): {(5 ^ 3) ^ 2} = {5 ^ (3 ^ 2)}")  # 结合律
    print()
    
    # 演示数组中的异或运算
    nums = [4, 1, 2, 1, 2]
    print(f"数组: {nums}")
    print("异或运算过程:")
    
    result = 0
    for i, num in enumerate(nums):
        result ^= num
        print(f"步骤 {i+1}: {result} ^ {num} = {result}")
    
    print(f"最终结果: {result}")
    print()
    
    # 为什么异或运算有效？
    print("为什么异或运算有效？")
    print("1. 相同的数字异或结果为0: a ^ a = 0")
    print("2. 任何数字与0异或结果不变: a ^ 0 = a")
    print("3. 异或运算满足交换律和结合律")
    print("4. 因此，成对出现的数字会相互抵消，只留下单独的数字")


if __name__ == "__main__":
    test_single_number()
    print("\n" + "="*50 + "\n")
    demonstrate_xor()
