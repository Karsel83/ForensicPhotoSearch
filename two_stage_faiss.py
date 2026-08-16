"""Two-stage FAISS retrieval for ForensicPhotoSearch.

Stage 1: FAISS retrieves a broad candidate set quickly.
Stage 2: exact cosine similarity re-ranks only those candidates.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

EMBEDDINGS_FILE = (
    ROOT
    / "data"
    / "embeddings"
    / "embeddings.npy"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "embeddings"
    / "metadata.json"
)

INDEX_FILE = (
    ROOT
    / "data"
    / "faiss"
    / "image.index"
)


def normalize(vectors):

    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / np.maximum(
        norms,
        1e-12
    )


class TwoStageImageSearch:

    def __init__(
        self,
        index_file=INDEX_FILE,
        embeddings_file=EMBEDDINGS_FILE,
        metadata_file=METADATA_FILE
    ):

        index_file = Path(
            index_file
        )

        embeddings_file = Path(
            embeddings_file
        )

        metadata_file = Path(
            metadata_file
        )

        # --------------------------------------------------
        # FAISS import
        # --------------------------------------------------

        try:

            import faiss

        except ImportError as error:

            raise RuntimeError(
                "Install FAISS first: "
                "python -m pip install faiss-cpu"
            ) from error

        # --------------------------------------------------
        # File check
        # --------------------------------------------------

        if not all(
            path.exists()
            for path in (
                index_file,
                embeddings_file,
                metadata_file
            )
        ):

            raise FileNotFoundError(

                "Missing index data.\n"
                f"FAISS index     : {index_file}\n"
                f"Embeddings      : {embeddings_file}\n"
                f"Metadata        : {metadata_file}\n\n"
                "Run:\n"
                "1. python build_database.py\n"
                "2. python build_embeddings.py\n"
                "3. python faiss_ann.py build"
            )

        # --------------------------------------------------
        # Load FAISS index
        # --------------------------------------------------

        self.index = faiss.read_index(
            str(index_file)
        )

        # --------------------------------------------------
        # Load embeddings
        # --------------------------------------------------

        self.embeddings = normalize(
            np.load(
                embeddings_file
            )
        )

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------

        self.metadata = json.loads(
            metadata_file.read_text(
                encoding="utf-8"
            )
        )

        # --------------------------------------------------
        # Consistency check
        # --------------------------------------------------

        if (
            self.index.ntotal
            != len(self.embeddings)
            or
            len(self.embeddings)
            != len(self.metadata)
        ):

            raise ValueError(
                "FAISS index, embeddings, and "
                "metadata must have the same length."
            )

    # ======================================================
    # Search
    # ======================================================

    def search(
        self,
        query_feature,
        candidate_k=1000,
        top_k=100
    ):

        # --------------------------------------------------
        # Query normalization
        # --------------------------------------------------

        query = normalize(
            np.asarray(
                query_feature,
                dtype=np.float32
            ).reshape(
                1,
                -1
            )
        )

        if query.shape[1] != (
            self.embeddings.shape[1]
        ):

            raise ValueError(
                "Query embedding dimension does not "
                "match the image embedding index."
            )

        # --------------------------------------------------
        # Search size
        # --------------------------------------------------

        candidate_k = min(
            max(
                int(candidate_k),
                int(top_k)
            ),
            self.index.ntotal
        )

        top_k = min(
            int(top_k),
            candidate_k
        )

        if candidate_k <= 0:
            return [], {
                "candidate_k": 0,
                "top_k": 0,
                "retrieval_ms": 0.0,
                "rerank_ms": 0.0,
                "total_ms": 0.0
            }

        # ==================================================
        # Stage 1
        # FAISS candidate retrieval
        # ==================================================

        retrieval_started = (
            time.perf_counter()
        )

        _, candidate_ids = (
            self.index.search(
                query,
                candidate_k
            )
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_started
        ) * 1000.0

        candidate_ids = (
            candidate_ids[0]
        )

        candidate_ids = (
            candidate_ids[
                candidate_ids >= 0
            ]
        )

        # ==================================================
        # Stage 2
        # Exact cosine re-ranking
        # ==================================================

        rerank_started = (
            time.perf_counter()
        )

        candidate_vectors = (
            self.embeddings[
                candidate_ids
            ]
        )

        exact_scores = (
            candidate_vectors
            @
            query[0]
        )

        order = np.argsort(
            -exact_scores,
            kind="stable"
        )[:top_k]

        rerank_ms = (
            time.perf_counter()
            - rerank_started
        ) * 1000.0

        # ==================================================
        # Result
        # ==================================================

        results = []

        for rank, position in enumerate(
            order,
            start=1
        ):

            vector_id = int(
                candidate_ids[position]
            )

            item = dict(
                self.metadata[
                    vector_id
                ]
            )

            item.update({

                "rank":
                    rank,

                "similarity":
                    round(
                        float(
                            exact_scores[
                                position
                            ]
                        ),
                        4
                    ),

                "vector_id":
                    vector_id
            })

            results.append(
                item
            )

        return results, {

            "candidate_k":
                int(
                    len(candidate_ids)
                ),

            "top_k":
                int(top_k),

            "retrieval_ms":
                round(
                    retrieval_ms,
                    6
                ),

            "rerank_ms":
                round(
                    rerank_ms,
                    6
                ),

            "total_ms":
                round(
                    retrieval_ms
                    + rerank_ms,
                    6
                )
        }


# ==========================================================
# Standalone test
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FAISS candidate retrieval "
            "plus exact Re-ID re-ranking."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Query person image path"
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100
    )

    args = parser.parse_args()

    from reid_model import PersonReID

    print(
        "[*] Loading Re-ID model..."
    )

    model = PersonReID()

    print(
        "[*] Extracting query feature..."
    )

    query_feature = model.extract(
        args.query
    )

    searcher = TwoStageImageSearch()

    results, metrics = (
        searcher.search(
            query_feature,
            args.candidate_k,
            args.top_k
        )
    )

    print()
    print(
        "[*] TWO-STAGE FAISS SEARCH"
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        )
    )

    for result in results:

        print(
            f"[{result['rank']}] "
            f"{result['filename']} "
            f"score={result['similarity']:.4f}"
        )

        print(
            f"    "
            f"candidate_id="
            f"{result['vector_id']} "
            f"crop="
            f"{result.get('crop')}"
        )


if __name__ == "__main__":
    main()