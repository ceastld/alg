"""
LeetCode 49. Group Anagrams

题目描述：
给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。
字母异位词是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

示例：
strs = ["eat","tea","tan","ate","nat","bat"]
输出：[["bat"],["nat","tan"],["ate","eat","tea"]]

数据范围：
- 1 <= strs.length <= 10^4
- 0 <= strs[i].length <= 100
- strs[i] 仅包含小写字母
"""

class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        from collections import defaultdict
        
        groups = defaultdict(list)
        
        for s in strs:
            # 使用排序后的字符串作为key
            key = ''.join(sorted(s))
            groups[key].append(s)
        
        return list(groups.values())
