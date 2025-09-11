d = "abcdefghijklmnopqrstuvwxyz"
s = input().strip()


def comp(c):
    try:
        return s.index(c)
    except:
        return 101


d = sorted(d, key=lambda x: comp(x))
s2 = input().strip()
o = [d[ord(c) - ord("a")] for c in s2]
print("".join(o))
