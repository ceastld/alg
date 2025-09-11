def encrypt_char(c):
    """Encrypt a single character"""
    if c.isalpha():
        # For letters: move one position forward and change case
        if c == 'Z':
            return 'a'
        elif c == 'z':
            return 'A'
        elif c.isupper():
            return chr(ord(c) + 1).lower()
        else:  # lowercase
            return chr(ord(c) + 1).upper()
    elif c.isdigit():
        # For digits: add 1, 9 becomes 0
        return str((int(c) + 1) % 10)
    return c

def decrypt_char(c):
    """Decrypt a single character"""
    if c.isalpha():
        # For letters: move one position backward and change case
        if c == 'a':
            return 'Z'
        elif c == 'A':
            return 'z'
        elif c.isupper():
            return chr(ord(c) - 1).lower()
        else:  # lowercase
            return chr(ord(c) - 1).upper()
    elif c.isdigit():
        # For digits: subtract 1, 0 becomes 9
        return str((int(c) - 1) % 10)
    return c

def encrypt_string(s):
    """Encrypt a string"""
    return ''.join(encrypt_char(c) for c in s)

def decrypt_string(s):
    """Decrypt a string"""
    return ''.join(decrypt_char(c) for c in s)

# Read input
s = input().strip()  # plaintext
t = input().strip()  # ciphertext

# Encrypt s and decrypt t
encrypted_s = encrypt_string(s)
decrypted_t = decrypt_string(t)

# Output results
print(encrypted_s)
print(decrypted_t)
