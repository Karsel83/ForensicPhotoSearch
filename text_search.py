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

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULT_FILE = (
    RESULT_DIR
    / "text_search_results.json"
)


# =========================================================
# Text Search
# =========================================================

class TextSearch:

    def __init__(
        self,
        model_path=MODEL_PATH,
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE
    ):

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

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
                f"Semantic FAISS index not found:\n"
                f"{index_path}"
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Semantic metadata not found:\n"
                f"{metadata_path}"
            )

        print(
            "[*] Loading semantic FAISS..."
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
            f"[+] Semantic FAISS loaded: "
            f"{self.index.ntotal} vectors"
        )

        print(
            f"[+] Embedding dimension: "
            f"{self.index.d}"
        )

    # =====================================================
    # Text Encoding
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
                    (
                        "Retrieve images relevant "
                        "to the user's query."
                    )
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

        vector = vector.astype(
            np.float32
        )

        # L2 normalize
        norm = np.linalg.norm(
            vector
        )

        if norm > 0:

            vector = (
                vector
                / norm
            )

        return vector

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query,
        top_k=20
    ):

        query = (
            str(query)
            .strip()
        )

        if not query:

            raise ValueError(
                "검색어가 비어 있습니다."
            )

        print()
        print(
            "=" * 70
        )

        print(
            "[*] TEXT SEARCH"
        )

        print(
            "=" * 70
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

        print(
            f"[*] Query embedding: "
            f"{query_vector.shape}"
        )

        # -------------------------------------------------
        # FAISS
        # -------------------------------------------------

        top_k = min(
            int(top_k),
            self.index.ntotal
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

            vector_id = int(
                vector_id
            )

            item = dict(
                self.metadata[
                    vector_id
                ]
            )

            item.update({

                "source_type":
                    "text_search",

                "query":
                    query,

                "rank":
                    rank,

                "vector_id":
                    vector_id,

                "similarity":
                    round(
                        float(score),
                        4
                    )
            })

            results.append(
                item
            )

        # -------------------------------------------------
        # Save result
        # -------------------------------------------------

        RESULT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output = {

            "query":
                query,

            "top_k":
                top_k,

            "index_size":
                self.index.ntotal,

            "embedding_dimension":
                self.index.d,

            "result_count":
                len(results),

            "results":
                results
        }

        with open(
            RESULT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=4
            )

        return results


# =========================================================
# CLI
# =========================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-VL + FAISS "
            "Text Search"
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help=(
            '검색어. 예: '
            '"빨간 패딩을 입은 남자"'
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20
    )

    args = parser.parse_args()

    searcher = TextSearch()

    results = searcher.search(
        args.query,
        args.top_k
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TEXT SEARCH RESULTS"
    )

    print(
        "=" * 70
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
            f"  Dataset    : "
            f"{result.get('dataset')}"
        )

        print(
            f"  Split      : "
            f"{result.get('split')}"
        )

        print(
            f"  Source     : "
            f"{result.get('source')}"
        )

        print()


if __name__ == "__main__":
    main()