a = input().strip()
# Extract only alphabetic characters from the string
v = [c.isalpha() for c in a]
sub = [c for c in a if c.isalpha()]
sub = sorted(sub,key=lambda x:x.lower())
j = 0
o = []
for i,k in enumerate(v):
    if k:
        o.append(sub[j])
        j += 1
    else:
        o.append(a[i])
print("".join(o))

