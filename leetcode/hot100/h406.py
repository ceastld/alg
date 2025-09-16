class Solution:
    def reconstructQueue(self, people: list[list[int]]) -> list[list[int]]:
        # 按身高降序，k值升序排序
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        for person in people:
            # 在k位置插入
            result.insert(person[1], person)
        
        return result
