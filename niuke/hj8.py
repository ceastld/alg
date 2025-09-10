
input()
import sys
m = {}
for line in sys.stdin:
    k,v =map(int,line.strip().split())
    if k in m:
        m[k]+=v
    else:
        m[k]=v
for k,v in sorted(m.items()):
    print(k,v)