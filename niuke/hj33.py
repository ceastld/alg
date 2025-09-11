def ip_to_int(ip: str) -> int:
    nums = [int(x) for x in ip.strip().split(".")]
    result = 0
    for num in nums:
        result = result * 256 + num
    return result


def int_to_ip(s: int) -> str:
    ip_parts = []
    for _ in range(4):
        ip_parts.append(s % 256)
        s = s // 256
    return ".".join(map(str, ip_parts[::-1]))


def main():
    print(ip_to_int(input().strip()))
    print(int_to_ip(int(input().strip())))

    
if __name__ == "__main__":
    main()
