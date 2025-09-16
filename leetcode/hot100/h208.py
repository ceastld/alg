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
