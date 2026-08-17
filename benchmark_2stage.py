import json
import time
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parent

EMBEDDINGS_FILE = (
    ROOT
    / "data"
    / "embeddings"
    / "embeddings.npy"
)

INDEX_FILE = (
    ROOT
    / "data"
    / "faiss"
    / "image.index"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "ann_benchmark"
    / "two_stage_results.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Settings
# =========================================================

QUERY_COUNT = 20

FINAL_TOP_K = 100

CANDIDATE_K_VALUES = [
    100,
    200,
    500,
    1000,
    2000,
    5000
]

RANDOM_SEED = 42


# =========================================================
# Normalize
# =========================================================

def normalize(
    vectors
):

    vectors = np.asarray(
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


# =========================================================
# Recall
# =========================================================

def recall_at_k(
    predicted,
    ground_truth,
    k
):

    total = 0
    correct = 0

    for pred, truth in zip(
        predicted,
        ground_truth
    ):

        truth_set = set(
            map(
                int,
                truth[:k]
            )
        )

        pred_set = set(
            map(
                int,
                pred[:k]
            )
        )

        correct += len(
            truth_set & pred_set
        )

        total += len(
            truth_set
        )

    if total == 0:
        return 0.0

    return (
        correct / total
    )


# =========================================================
# Main
# =========================================================

def main():

    print(
        "[*] Loading embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    embeddings = normalize(
        embeddings
    )

    print(
        f"[*] Embeddings: "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------
    # Load FAISS
    # -----------------------------------------------------

    print(
        "[*] Loading FAISS index..."
    )

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"[*] FAISS vectors: "
        f"{index.ntotal}"
    )

    # -----------------------------------------------------
    # Queries
    # -----------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    query_indices = rng.choice(
        len(embeddings),
        size=min(
            QUERY_COUNT,
            len(embeddings)
        ),
        replace=False
    )

    queries = embeddings[
        query_indices
    ]

    # -----------------------------------------------------
    # Exact Ground Truth
    # -----------------------------------------------------

    print(
        "[*] Building exact ground truth..."
    )

    exact_index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    exact_index.add(
        embeddings
    )

    _, ground_truth = (
        exact_index.search(
            queries,
            FINAL_TOP_K
        )
    )

    results = []

    # =====================================================
    # Candidate K experiments
    # =====================================================

    for candidate_k in (
        CANDIDATE_K_VALUES
    ):

        candidate_k = min(
            candidate_k,
            len(embeddings)
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"Candidate K = "
            f"{candidate_k}"
        )

        # -------------------------------------------------
        # Stage 1
        # -------------------------------------------------

        stage1_times = []

        stage1_results = []

        # -------------------------------------------------
        # Stage 2
        # -------------------------------------------------

        stage2_times = []

        final_results = []

        for query in queries:

            query = query.reshape(
                1,
                -1
            )

            # =============================================
            # STAGE 1
            # ANN Candidate Retrieval
            # =============================================

            start = (
                time.perf_counter()
            )

            _, candidate_ids = (
                index.search(
                    query,
                    candidate_k
                )
            )

            stage1_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            candidate_ids = (
                candidate_ids[0]
            )

            candidate_ids = (
                candidate_ids[
                    candidate_ids >= 0
                ]
            )

            stage1_times.append(
                stage1_ms
            )

            # Save the whole candidate set
            stage1_results.append(
                candidate_ids
            )

            # =============================================
            # STAGE 2
            # Exact cosine re-ranking
            # =============================================

            start = (
                time.perf_counter()
            )

            candidate_vectors = (
                embeddings[
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
            )[:FINAL_TOP_K]

            reranked_ids = (
                candidate_ids[
                    order
                ]
            )

            stage2_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            stage2_times.append(
                stage2_ms
            )

            final_results.append(
                reranked_ids
            )

        # -------------------------------------------------
        # Stage 1 Recall
        #
        # "Exact Top-100 중 몇 개가 ANN 후보군에
        #  들어왔는가?"
        # -------------------------------------------------

        stage1_recall_1 = recall_at_k(
            stage1_results,
            ground_truth,
            1
        )

        stage1_recall_10 = recall_at_k(
            stage1_results,
            ground_truth,
            10
        )

        stage1_recall_100 = recall_at_k(
            stage1_results,
            ground_truth,
            100
        )

        # -------------------------------------------------
        # Stage 2 / Final Recall
        # -------------------------------------------------

        final_recall_1 = recall_at_k(
            final_results,
            ground_truth,
            1
        )

        final_recall_10 = recall_at_k(
            final_results,
            ground_truth,
            10
        )

        final_recall_100 = recall_at_k(
            final_results,
            ground_truth,
            100
        )

        # -------------------------------------------------
        # Timing
        # -------------------------------------------------

        stage1_mean = float(
            np.mean(
                stage1_times
            )
        )

        stage2_mean = float(
            np.mean(
                stage2_times
            )
        )

        total_mean = (
            stage1_mean
            + stage2_mean
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        result = {

            "candidate_k":
                candidate_k,

            "final_top_k":
                FINAL_TOP_K,

            "stage1_ms_mean":
                round(
                    stage1_mean,
                    6
                ),

            "stage2_ms_mean":
                round(
                    stage2_mean,
                    6
                ),

            "total_ms_mean":
                round(
                    total_mean,
                    6
                ),

            "stage1_recall@1":
                round(
                    stage1_recall_1,
                    6
                ),

            "stage1_recall@10":
                round(
                    stage1_recall_10,
                    6
                ),

            "stage1_recall@100":
                round(
                    stage1_recall_100,
                    6
                ),

            "final_recall@1":
                round(
                    final_recall_1,
                    6
                ),

            "final_recall@10":
                round(
                    final_recall_10,
                    6
                ),

            "final_recall@100":
                round(
                    final_recall_100,
                    6
                )
        }

        results.append(
            result
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

    # =====================================================
    # Save
    # =====================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "dataset_size":
                    len(embeddings),

                "query_count":
                    len(queries),

                "final_top_k":
                    FINAL_TOP_K,

                "candidate_k_values":
                    CANDIDATE_K_VALUES,

                "results":
                    results
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        "=" * 60
    )

    print(
        "[*] 2-STAGE BENCHMARK COMPLETE"
    )

    print(
        f"[*] Saved: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()