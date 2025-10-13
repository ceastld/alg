"""
406. 根据身高重建队列
假设有打乱顺序的一群站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

题目链接：https://leetcode.cn/problems/queue-reconstruction-by-height/

示例 1:
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。

示例 2:
输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]

提示：
- 1 <= people.length <= 2000
- 0 <= hi <= 10^6
- 0 <= ki < people.length
- 题目数据确保队列可以被重建
"""

from typing import List


class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        result = []
        for p in people:
            result.insert(p[1], p)
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    result1 = solution.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])
    expected1 = [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    assert result1 == expected1
    
    # 测试用例2
    result2 = solution.reconstructQueue([[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]])
    expected2 = [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
    assert result2 == expected2
    
    # 测试用例3
    result3 = solution.reconstructQueue([[1,0]])
    expected3 = [[1,0]]
    assert result3 == expected3
    
    # 测试用例4
    result4 = solution.reconstructQueue([[2,0],[1,1]])
    expected4 = [[2,0],[1,1]]
    assert result4 == expected4
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
