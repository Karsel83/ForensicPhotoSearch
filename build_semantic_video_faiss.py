from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "semantic_video"
    / "embeddings.npy"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "semantic_video"
)

INDEX_FILE = (
    INDEX_DIR
    / "video.index"
)


def main():

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = np.load(
        EMBEDDING_FILE
    ).astype(
        np.float32
    )

    print(
        f"[*] Embeddings: "
        f"{embeddings.shape}"
    )

    dimension = (
        embeddings.shape[1]
    )

    # L2 normalized embeddings
    # + inner product
    # = cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print()
    print(
        "=" * 60
    )

    print(
        "[*] Semantic video FAISS index created"
    )

    print(
        f"[*] Vectors: "
        f"{index.ntotal}"
    )

    print(
        f"[*] Dimension: "
        f"{dimension}"
    )

    print(
        f"[*] Saved: "
        f"{INDEX_FILE}"
    )


if __name__ == "__main__":
    main()