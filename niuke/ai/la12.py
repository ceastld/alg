import numpy as np

def cosine_similarity(v1, v2):
    value = np.dot(v1,v2)/np.sqrt(np.dot(v1,v1)*np.dot(v2,v2))
    return round(value,3)


if __name__ == "__main__":
    v1 = np.array(eval(input()))
    v2 = np.array(eval(input()))
    print(cosine_similarity(v1, v2))