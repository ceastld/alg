"""
860. 柠檬水找零
在柠檬水摊上，每一杯柠檬水的售价为 5 美元。顾客排队购买柠檬水，按账单 bills 支付的顺序，一次购买一杯。

每位顾客只买一杯柠檬水，然后向你支付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。

注意，一开始你手头没有任何零钱。

给你一个整数数组 bills ，其中 bills[i] 是第 i 位顾客付的账。如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

题目链接：https://leetcode.cn/problems/lemonade-change/

示例 1:
输入：bills = [5,5,5,10,20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找零一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true。

示例 2:
输入：bills = [5,5,10,10,20]
输出：false
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以我们输出 false。

提示：
- 1 <= bills.length <= 10^5
- bills[i] 不是 5 就是 10 或 20
"""

from typing import List


class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        c5 = 0
        c10 = 0
        for bill in bills:
            if bill == 5:
                c5 += 1
            elif bill == 10:
                c10 += 1
                if c5 > 0:
                    c5 -= 1
                else:
                    return False
            elif bill == 20:
                if c10 > 0 and c5 > 0:
                    c10 -= 1
                    c5 -= 1
                elif c5 >= 3:
                    c5 -= 3
                else:
                    return False
        return True


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    assert solution.lemonadeChange([5, 5, 5, 10, 20]) == True

    # 测试用例2
    assert solution.lemonadeChange([5, 5, 10, 10, 20]) == False

    # 测试用例3
    assert solution.lemonadeChange([5, 5, 10]) == True

    # 测试用例4
    assert solution.lemonadeChange([10, 10]) == False

    # 测试用例5
    assert solution.lemonadeChange([5, 5, 5, 10, 5, 20, 5, 10, 5, 20]) == True

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
