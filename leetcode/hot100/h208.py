"""
LeetCode 208. Implement Trie (Prefix Tree)

题目描述：
Trie（发音类似"try"）或者说前缀树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。
请你实现Trie类：
- Trie()初始化前缀树对象
- void insert(String word)向前缀树中插入字符串word
- boolean search(String word)如果字符串word在前缀树中，返回true（即，在检索之前已经插入）；否则，返回false
- boolean startsWith(String prefix)如果之前已经插入的字符串word的前缀之一为prefix，返回true；否则，返回false

示例：
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回True
trie.search("app");     // 返回False
trie.startsWith("app"); // 返回True
trie.insert("app");
trie.search("app");     // 返回True

数据范围：
- 1 <= word.length, prefix.length <= 2000
- word和prefix仅由小写英文字母组成
- insert、search和startsWith调用次数总计不超过3 * 10^4次
"""

class TrieNode:
    """Trie node class"""
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_end_of_word = False  # Flag to mark end of word

class Trie(object):
    """Implement Trie (Prefix Tree)"""
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
    
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Alternative implementation using nested dictionaries (more concise)
class TrieDict(object):
    """Alternative Trie implementation using nested dictionaries"""
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
    
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True  # Mark end of word
    
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.trie
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '#' in node
    
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.trie
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True

# Test cases
if __name__ == "__main__":
    # Test Trie class
    trie = Trie()
    
    # Insert words
    trie.insert("apple")
    trie.insert("app")
    trie.insert("application")
    
    # Test search
    print(trie.search("app"))      # True
    print(trie.search("apple"))    # True
    print(trie.search("appl"))     # False (not a complete word)
    print(trie.search("xyz"))      # False
    
    # Test startsWith
    print(trie.startsWith("app"))  # True
    print(trie.startsWith("ap"))   # True
    print(trie.startsWith("xyz"))  # False
    
    print("\n" + "="*50 + "\n")
    
    # Test TrieDict class
    trie_dict = TrieDict()
    
    # Insert words
    trie_dict.insert("apple")
    trie_dict.insert("app")
    trie_dict.insert("application")
    
    # Test search
    print(trie_dict.search("app"))      # True
    print(trie_dict.search("apple"))    # True
    print(trie_dict.search("appl"))     # False
    print(trie_dict.search("xyz"))      # False
    
    # Test startsWith
    print(trie_dict.startsWith("app"))  # True
    print(trie_dict.startsWith("ap"))   # True
    print(trie_dict.startsWith("xyz"))  # False
