"""
LeetCode 406. Queue Reconstruction by Height

题目描述：
假设有打乱顺序的一群人站成一个队列，数组people表示队列中一些人的属性（不一定按顺序）。每个people[i] = [hi, ki]表示第i个人的身高为hi，前面正好有ki个身高大于或等于hi的人。
请你重新构造并返回输入数组people所表示的队列。返回的队列应该格式化为数组queue，其中queue[j] = [hj, kj]是队列中第j个人的属性（queue[0]是排在队列前面的人）。

示例：
people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]

数据范围：
- 1 <= people.length <= 2000
- 0 <= hi <= 10^6
- 0 <= ki < people.length
- 题目数据确保队列可以被重建
"""

class Solution:
    def reconstructQueue(self, people: list[list[int]]) -> list[list[int]]:
        # 按身高降序，k值升序排序
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        for person in people:
            # 在k位置插入
            result.insert(person[1], person)
        
        return result
