class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        if len(s) < len(p):
            return []
        
        result = []
        p_count = [0] * 26
        s_count = [0] * 26
        
        # 统计p中每个字符的频次
        for char in p:
            p_count[ord(char) - ord('a')] += 1
        
        # 滑动窗口
        for i in range(len(s)):
            # 添加新字符
            s_count[ord(s[i]) - ord('a')] += 1
            
            # 移除窗口外的字符
            if i >= len(p):
                s_count[ord(s[i - len(p)]) - ord('a')] -= 1
            
            # 检查是否匹配
            if s_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
