def solve():
    # Initialize counters: A, B, C, D, E, error, private
    cnt = [0] * 7
    
    try:
        while True:
            line = input()
            if not line:
                break
            
            try:
                ip_str, mask_str = line.split('~')
                
                # Parse IP and mask
                ip = [int(x) for x in ip_str.split('.')]
                mask = [int(x) for x in mask_str.split('.')]
                
                # Check format validity
                if len(ip) != 4 or len(mask) != 4 or any(x < 0 or x > 255 for x in ip + mask):
                    cnt[5] += 1  # error
                    continue
                
                # Skip special IPs
                if ip[0] == 0 or ip[0] == 127:
                    continue
                
                # Check subnet mask validity
                binary = ''.join(format(x, '08b') for x in mask)
                if binary == '1' * 32 or binary == '0' * 32 or '01' in binary:
                    cnt[5] += 1  # error
                    continue
                
                # Count IP class
                if 1 <= ip[0] <= 127:
                    cnt[0] += 1  # A
                elif 128 <= ip[0] <= 191:
                    cnt[1] += 1  # B
                elif 192 <= ip[0] <= 223:
                    cnt[2] += 1  # C
                elif 224 <= ip[0] <= 239:
                    cnt[3] += 1  # D
                elif 240 <= ip[0] <= 255:
                    cnt[4] += 1  # E
                
                # Count private IP
                if (ip[0] == 10 or 
                    (ip[0] == 172 and 16 <= ip[1] <= 31) or 
                    (ip[0] == 192 and ip[1] == 168)):
                    cnt[6] += 1  # private
                    
            except:
                cnt[5] += 1  # error
                
    except EOFError:
        pass
    
    print(' '.join(map(str, cnt)))


if __name__ == "__main__":
    solve()
