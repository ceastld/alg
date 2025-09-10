I = input().split()
ss = I[1:-2]
x = I[-2]
k = int(I[-1])
d = []
key_x = "".join(sorted(x))
for s in ss:
    key = "".join(sorted(s))
    if key == key_x and s != x:
        d.append(s)

res = sorted(d)
print(len(res))
if len(res) > k-1:
    print(res[k-1])