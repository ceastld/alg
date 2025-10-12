"""
93. 复原IP地址
有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导零），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。

给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。

题目链接：https://leetcode.cn/problems/restore-ip-addresses/

示例 1:
输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]

示例 2:
输入：s = "0000"
输出：["0.0.0.0"]

示例 3:
输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]

提示：
- 1 <= s.length <= 20
- s 仅由数字组成
"""

from typing import List


class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(ip):
            if len(ip) > 3:
                return False
            val = int(ip)
            if val > 255:
                return False
            if ip[0] == "0" and len(ip) > 1:
                return False
            return True

        def dfs(start, ip):
            if len(ip) == 4:
                if start == len(s):
                    result.append(".".join(ip))
                return
            
            for end in range(start, min(start + 3, len(s))):
                select = s[start : end + 1]
                if is_valid(select):
                    dfs(end + 1, ip + [select])

        result = []
        dfs(0, [])
        return result


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    s = "25525511135"
    result = solution.restoreIpAddresses(s)
    expected = ["255.255.11.135", "255.255.111.35"]
    assert set(result) == set(expected)

    # 测试用例2
    s = "0000"
    result = solution.restoreIpAddresses(s)
    expected = ["0.0.0.0"]
    assert result == expected

    # 测试用例3
    s = "101023"
    result = solution.restoreIpAddresses(s)
    expected = ["1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3"]
    assert set(result) == set(expected)

    # 测试用例4
    s = "1111"
    result = solution.restoreIpAddresses(s)
    expected = ["1.1.1.1"]
    assert result == expected

    # 测试用例5
    s = "010010"
    result = solution.restoreIpAddresses(s)
    expected = ["0.10.0.10", "0.100.1.0"]
    assert set(result) == set(expected)

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
