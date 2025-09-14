n = int(input().strip())
matrix = [[0 for _ in range(n)] for _ in range(n)]
count = 1
for i in range(0, n):
    for j in range(0, i + 1):
        matrix[i - j][j] = count
        count += 1

for row in matrix:
    print(" ".join(map(str, [x for x in row if x != 0])))
