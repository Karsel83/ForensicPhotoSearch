import json
from pathlib import Path

import numpy as np


def save_database(data, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


def load_database(input_path):

    input_path = Path(input_path)

    if not input_path.exists():
        return []

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


class ImageEmbeddingIndex:
    """Persistent dense-vector index for already extracted person embeddings."""

    def __init__(self, embeddings_path, metadata_path):
        embeddings_path = Path(embeddings_path)
        metadata_path = Path(metadata_path)

        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Image embedding index not found. Run build_database.py and "
                "build_embeddings.py before searching."
            )

        self.embeddings = np.load(embeddings_path).astype(np.float32)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.embeddings.ndim != 2:
            raise ValueError("Image embedding index must be a 2-D array.")

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                "Embedding index and metadata contain different numbers of entries."
            )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.embeddings / np.maximum(norms, 1e-12)

    def search(self, query_feature, top_k=50):
        query = np.asarray(query_feature, dtype=np.float32).reshape(-1)

        if query.shape[0] != self.normalized_embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match the image index."
            )

        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = self.normalized_embeddings @ query
        limit = min(max(int(top_k), 0), len(scores))

        if limit == 0:
            return []

        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            (self.metadata[int(index)], float(scores[int(index)]))
            for index in order
        ]
