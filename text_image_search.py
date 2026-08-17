from pathlib import Path
import json
import sys

import faiss
import numpy as np
import torch


# =========================================================
# Project Root
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

QWEN_ROOT = (
    PROJECT_ROOT
    / "Qwen3-VL-Embedding"
)

if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(QWEN_ROOT)
    )

from src.models.qwen3_vl_embedding import (
    Qwen3VLEmbedder
)


# =========================================================
# Paths
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-vl"
    / "Qwen3-VL-Embedding-2B"
)

SEMANTIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "semantic"
)

INDEX_FILE = (
    SEMANTIC_DIR
    / "image.index"
)

METADATA_FILE = (
    SEMANTIC_DIR
    / "metadata.json"
)


# =========================================================
# Text → Image Search
# =========================================================

class TextImageSearch:

    def __init__(
        self,
        model_path=MODEL_PATH,
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE
    ):

        print(
            "[*] Loading Qwen3-VL "
            "Embedding model..."
        )

        self.model = (
            Qwen3VLEmbedder(
                model_name_or_path=
                    str(model_path)
            )
        )

        print(
            "[+] Qwen3-VL "
            "Embedding loaded"
        )

        # -------------------------------------------------
        # FAISS
        # -------------------------------------------------

        index_path = Path(
            index_path
        )

        metadata_path = Path(
            metadata_path
        )

        if not index_path.exists():

            raise FileNotFoundError(
                f"Semantic FAISS index not found: "
                f"{index_path}\n"
                "Run build_semantic_embeddings.py "
                "and build_semantic_faiss.py first."
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Semantic metadata not found: "
                f"{metadata_path}"
            )

        print(
            "[*] Loading semantic FAISS index..."
        )

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(
                f
            )

        if (
            self.index.ntotal
            != len(self.metadata)
        ):

            raise ValueError(
                "FAISS index and metadata "
                "length do not match."
            )

        print(
            f"[+] Semantic index loaded: "
            f"{self.index.ntotal} vectors"
        )

    # =====================================================
    # Text Embedding
    # =====================================================

    def encode_text(
        self,
        text
    ):

        inputs = [
            {
                "text":
                    text,

                "instruction":
                    "Retrieve images relevant to the user's query."
            }
        ]

        embedding = (
            self.model.process(
                inputs
            )
        )

        if torch.is_tensor(
            embedding
        ):

            vector = (
                embedding[0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        else:

            vector = np.asarray(
                embedding[0],
                dtype=np.float32
            )

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        vector = vector.astype(
            np.float32
        )

        norm = np.linalg.norm(
            vector
        )

        if norm > 0:

            vector = (
                vector / norm
            )

        return vector

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query,
        top_k=10
    ):

        print()
        print(
            "=" * 60
        )

        print(
            "[*] TEXT → IMAGE "
            "FAISS SEARCH"
        )

        print(
            "=" * 60
        )

        print(
            f"[*] Query: {query}"
        )

        # -------------------------------------------------
        # Text embedding
        # -------------------------------------------------

        query_vector = (
            self.encode_text(
                query
            )
        )

        query_vector = (
            query_vector
            .reshape(
                1,
                -1
            )
            .astype(
                np.float32
            )
        )

        if (
            query_vector.shape[1]
            != self.index.d
        ):

            raise ValueError(
                "Query embedding dimension "
                "does not match semantic index."
            )

        # -------------------------------------------------
        # FAISS search
        # -------------------------------------------------

        top_k = min(
            int(top_k),
            self.index.ntotal
        )

        scores, ids = (
            self.index.search(
                query_vector,
                top_k
            )
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        results = []

        for rank, (
            score,
            vector_id
        ) in enumerate(
            zip(
                scores[0],
                ids[0]
            ),
            start=1
        ):

            if vector_id < 0:
                continue

            item = dict(
                self.metadata[
                    int(vector_id)
                ]
            )

            item.update({

                "source_type":
                    "text_image",

                "query":
                    query,

                "similarity":
                    round(
                        float(score),
                        4
                    ),

                "vector_id":
                    int(vector_id),

                "rank":
                    rank
            })

            results.append(
                item
            )

        return results


# =========================================================
# Standalone Test
# =========================================================

def main():

    searcher = (
        TextImageSearch()
    )

    query = (
        "빨간 패딩을 입은 남자"
    )

    results = (
        searcher.search(
            query,
            top_k=10
        )
    )

    print()
    print(
        "=" * 60
    )

    print(
        "TOP RESULTS"
    )

    print(
        "=" * 60
    )

    for result in results:

        print(
            f"[Rank {result['rank']}] "
            f"{result.get('filename')}"
        )

        print(
            f"  Similarity : "
            f"{result['similarity']}"
        )

        print(
            f"  Source     : "
            f"{result.get('source')}"
        )

        print(
            f"  Vector ID  : "
            f"{result['vector_id']}"
        )


if __name__ == "__main__":
    main()