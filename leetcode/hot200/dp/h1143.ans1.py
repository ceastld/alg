from collections import defaultdict
from bisect import bisect_left

def LCS(s1, s2):
    m = len(s1)
    hashmap = defaultdict(list)
    for i in range(m - 1, -1, -1):
        hashmap[s1[i]].append(i)
    candidates = []
    for c in s2:
        if c in hashmap:
            candidates.extend(hashmap[c])

    return LIS(candidates)


def LIS(candidates):
    stack = []
    for num in candidates:
        idx = bisect_left(stack, num)
        if idx < len(stack):
            stack[idx] = num
        else:
            stack.append(num)
    return len(stack)

class Solution:
    def longestCommonSubsequence(self, a: str, b: str) -> int:
        if a in b: return len(a)
        if b in a: return len(b)
        return LCS(a,b)