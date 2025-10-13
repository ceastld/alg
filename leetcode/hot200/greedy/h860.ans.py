"""
860. 柠檬水找零 - 标准答案
"""
from typing import List


class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 维护5美元和10美元的数量
        2. 对于5美元，直接收取
        3. 对于10美元，找零一张5美元
        4. 对于20美元，优先找零一张10美元和一张5美元，否则找零三张5美元
        5. 贪心策略：优先使用10美元找零，因为5美元更灵活
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        five = 0  # 5美元数量
        ten = 0    # 10美元数量
        
        for bill in bills:
            if bill == 5:
                # 5美元直接收取
                five += 1
            elif bill == 10:
                # 10美元需要找零一张5美元
                if five > 0:
                    five -= 1
                    ten += 1
                else:
                    return False
            else:  # bill == 20
                # 20美元需要找零15美元
                if ten > 0 and five > 0:
                    # 优先找零一张10美元和一张5美元
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    # 否则找零三张5美元
                    five -= 3
                else:
                    return False
        
        return True
    
    def lemonadeChange_alternative(self, bills: List[int]) -> bool:
        """
        替代解法：贪心算法（使用字典）
        
        解题思路：
        1. 使用字典维护各种面额的数量
        2. 根据面额进行相应的找零操作
        3. 贪心策略：优先使用大面额找零
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        change = {5: 0, 10: 0, 20: 0}
        
        for bill in bills:
            change[bill] += 1
            
            if bill == 10:
                # 10美元需要找零一张5美元
                if change[5] > 0:
                    change[5] -= 1
                else:
                    return False
            elif bill == 20:
                # 20美元需要找零15美元
                if change[10] > 0 and change[5] > 0:
                    # 优先找零一张10美元和一张5美元
                    change[10] -= 1
                    change[5] -= 1
                elif change[5] >= 3:
                    # 否则找零三张5美元
                    change[5] -= 3
                else:
                    return False
        
        return True
    
    def lemonadeChange_optimized(self, bills: List[int]) -> bool:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 只维护5美元和10美元的数量
        2. 对于20美元，优先找零一张10美元和一张5美元
        3. 否则找零三张5美元
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        five = 0
        ten = 0
        
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five > 0:
                    five -= 1
                    ten += 1
                else:
                    return False
            else:  # bill == 20
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
        
        return True
    
    def lemonadeChange_detailed(self, bills: List[int]) -> bool:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 维护5美元和10美元的数量
        2. 对于每个账单，进行相应的找零操作
        3. 贪心策略：优先使用10美元找零，因为5美元更灵活
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        five = 0  # 5美元数量
        ten = 0   # 10美元数量
        
        for bill in bills:
            if bill == 5:
                # 5美元直接收取，不需要找零
                five += 1
            elif bill == 10:
                # 10美元需要找零一张5美元
                if five > 0:
                    five -= 1
                    ten += 1
                else:
                    # 没有5美元找零，返回false
                    return False
            else:  # bill == 20
                # 20美元需要找零15美元
                if ten > 0 and five > 0:
                    # 优先找零一张10美元和一张5美元
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    # 否则找零三张5美元
                    five -= 3
                else:
                    # 无法找零，返回false
                    return False
        
        return True
    
    def lemonadeChange_brute_force(self, bills: List[int]) -> bool:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的找零方案
        2. 检查是否能成功找零
        
        时间复杂度：O(3^n)
        空间复杂度：O(n)
        """
        def backtrack(index, five, ten):
            if index == len(bills):
                return True
            
            bill = bills[index]
            
            if bill == 5:
                return backtrack(index + 1, five + 1, ten)
            elif bill == 10:
                if five > 0:
                    return backtrack(index + 1, five - 1, ten + 1)
                else:
                    return False
            else:  # bill == 20
                # 尝试找零一张10美元和一张5美元
                if ten > 0 and five > 0:
                    if backtrack(index + 1, five - 1, ten - 1):
                        return True
                
                # 尝试找零三张5美元
                if five >= 3:
                    if backtrack(index + 1, five - 3, ten):
                        return True
                
                return False
        
        return backtrack(0, 0, 0)
    
    def lemonadeChange_step_by_step(self, bills: List[int]) -> bool:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：处理5美元和10美元
        2. 第二步：处理20美元
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        five = 0
        ten = 0
        
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five > 0:
                    five -= 1
                    ten += 1
                else:
                    return False
            else:  # bill == 20
                # 优先找零一张10美元和一张5美元
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
        
        return True


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.lemonadeChange([5,5,5,10,20]) == True
    
    # 测试用例2
    assert solution.lemonadeChange([5,5,10,10,20]) == False
    
    # 测试用例3
    assert solution.lemonadeChange([5,5,10]) == True
    
    # 测试用例4
    assert solution.lemonadeChange([10,10]) == False
    
    # 测试用例5
    assert solution.lemonadeChange([5,5,5,10,5,20,5,10,5,20]) == True
    
    # 测试用例6：边界情况
    assert solution.lemonadeChange([5]) == True
    assert solution.lemonadeChange([10]) == False
    assert solution.lemonadeChange([20]) == False
    
    # 测试用例7：全5美元
    assert solution.lemonadeChange([5,5,5,5,5]) == True
    
    # 测试用例8：全10美元
    assert solution.lemonadeChange([10,10,10]) == False
    
    # 测试用例9：全20美元
    assert solution.lemonadeChange([20,20,20]) == False
    
    # 测试用例10：复杂情况
    assert solution.lemonadeChange([5,5,10,20,5,5,5,5,5,5]) == True
    assert solution.lemonadeChange([5,5,10,20,5,5,5,5,5,5,20]) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
