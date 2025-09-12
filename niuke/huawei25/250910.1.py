import math
from typing import List, Dict, Tuple
from collections import Counter


def calculate_tf_idf(documents: List[List[str]]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculate TF-IDF for documents in the window.
    
    Args:
        documents: List of tokenized documents
        
    Returns:
        Tuple of (idf_scores, tf_idf_vectors)
    """
    N = len(documents)
    
    # Calculate document frequency (df) for each term
    term_doc_count = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_count[term] += 1
    
    # Calculate IDF using smooth formula: IDF(x) = log((N+1)/(df(x)+1)) + 1
    idf_scores = {}
    for term, df in term_doc_count.items():
        idf_scores[term] = math.log((N + 1) / (df + 1)) + 1
    
    # Calculate TF-IDF vectors for each document
    tf_idf_vectors = []
    for doc in documents:
        tf = Counter(doc)
        tf_idf_vector = {}
        for term, tf_count in tf.items():
            tf_idf_vector[term] = tf_count * idf_scores[term]
        tf_idf_vectors.append(tf_idf_vector)
    
    return idf_scores, tf_idf_vectors


def calculate_weighted_cosine_similarity(
    query_vector: Dict[str, float], 
    doc_vector: Dict[str, float], 
    weight: float
) -> float:
    """
    Calculate weighted cosine similarity between query and document.
    
    Args:
        query_vector: TF-IDF vector of query
        doc_vector: TF-IDF vector of document
        weight: Time weight for the document
        
    Returns:
        Weighted cosine similarity score
    """
    # Apply time weight to document vector
    weighted_doc_vector = {term: score * weight for term, score in doc_vector.items()}
    
    # Calculate dot product
    dot_product = 0.0
    for term in query_vector:
        if term in weighted_doc_vector:
            dot_product += query_vector[term] * weighted_doc_vector[term]
    
    # Calculate norms
    query_norm = math.sqrt(sum(score ** 2 for score in query_vector.values()))
    doc_norm = math.sqrt(sum(score ** 2 for score in weighted_doc_vector.values()))
    
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    
    return dot_product / (query_norm * doc_norm)


def find_most_similar_document(
    documents: List[str], 
    query: str, 
    window_size: int, 
    time_point: int
) -> int:
    """
    Find the most similar document in the time window.
    
    Args:
        documents: List of all documents
        query: Query string
        window_size: Size of time window K
        time_point: Current time point t
        
    Returns:
        Document index or -1 if no document meets threshold
    """
    # Get documents in the window [t-K+1, t]
    start_idx = max(0, time_point - window_size + 1)
    window_docs = documents[start_idx:time_point + 1]
    window_indices = list(range(start_idx, time_point + 1))
    
    if not window_docs:
        return -1
    
    # Tokenize documents and query
    tokenized_docs = [doc.lower().split() for doc in window_docs]
    query_tokens = query.lower().split()
    
    # Calculate TF-IDF for window documents
    idf_scores, tf_idf_vectors = calculate_tf_idf(tokenized_docs)
    
    # Calculate query TF-IDF vector
    query_tf = Counter(query_tokens)
    query_vector = {}
    for term, tf_count in query_tf.items():
        if term in idf_scores:
            query_vector[term] = tf_count * idf_scores[term]
    
    if not query_vector:
        return -1
    
    # Calculate similarities with time weights
    similarities = []
    for i, (doc_vector, doc_idx) in enumerate(zip(tf_idf_vectors, window_indices)):
        # Time weight: (K-j+1)/K where j is position in window (1-indexed)
        # j = i + 1 (convert 0-indexed to 1-indexed)
        j = i + 1
        time_weight = (window_size - j + 1) / window_size
        similarity = calculate_weighted_cosine_similarity(query_vector, doc_vector, time_weight)
        similarities.append((similarity, doc_idx))
    
    # Filter by threshold and find best match
    valid_similarities = [(sim, idx) for sim, idx in similarities if sim >= 0.6]
    
    if not valid_similarities:
        return -1
    
    # Sort by similarity (descending) and then by document index (ascending for earliest)
    valid_similarities.sort(key=lambda x: (-x[0], x[1]))
    
    return valid_similarities[0][1]


def solve():
    """Main solution function."""
    # Read input
    n = int(input())
    documents = []
    for _ in range(n):
        documents.append(input().strip())
    
    k = int(input())
    p = int(input())
    
    queries = []
    for _ in range(p):
        line = input().strip().split(' ', 1)
        t = int(line[0])
        q = line[1]
        queries.append((t, q))
    
    # Process each query
    results = []
    for t, q in queries:
        result = find_most_similar_document(documents, q, k, t)
        results.append(str(result))
    
    # Output results
    print(' '.join(results))


if __name__ == "__main__":
    solve()
