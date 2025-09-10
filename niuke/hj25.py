I = list(map(str.strip,input().split()[1:]))
R = map(str.strip,input().split()[1:])
R = sorted(set(R), key=lambda x:int(x))
output = []
for r in R:
    t = []
    for id,i in enumerate(I):
        if i.find(r) != -1:
            t.append(id)
            t.append(i)
    if t:
        output.append(r)
        output.append(len(t)//2)
        output.extend(t)

print(len(output),end=" ")
for o in output:
    print(o,end=" ")
print()

