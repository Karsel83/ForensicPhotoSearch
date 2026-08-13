import numpy as np


def cosine_similarity(vector_a, vector_b):

    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )