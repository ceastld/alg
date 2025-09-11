s = input().strip()
print("".join(sorted(s,key=lambda x:ord(x))))