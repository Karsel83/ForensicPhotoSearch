from pathlib import Path
import json
import time

from text_image_search import TextImageSearch
from semantic_video_search import SemanticVideoSearch
from score_fusion import fuse_modalities


PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "unified_semantic_results.json"
)


IMAGE_TOP_K = 10
IMAGE_WEIGHT = 0.5
VIDEO_WEIGHT = 0.5


class UnifiedSemanticSearch:

    def __init__(
        self
    ):

        print(
            "[*] Initializing Unified "
            "Semantic Search..."
        )

        print(
            "[*] Initializing semantic "
            "image search..."
        )

        self.image_search = (
            TextImageSearch()
        )

        print(
            "[*] Initializing semantic "
            "video search..."
        )

        self.video_search = (
            SemanticVideoSearch()
        )

        print(
            "[+] Unified Semantic Search ready"
        )

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query
    ):

        print()
        print("=" * 70)
        print(
            "UNIFIED SEMANTIC SEARCH"
        )
        print("=" * 70)

        print(
            f"[*] Query: {query}"
        )

        started = (
            time.perf_counter()
        )

        # -------------------------------------------------
        # Image
        # -------------------------------------------------

        print()
        print(
            "[1/2] Semantic image search..."
        )

        image_results = (
            self.image_search.search(
                query,
                top_k=IMAGE_TOP_K
            )
        )

        # -------------------------------------------------
        # Video
        # -------------------------------------------------

        print()
        print(
            "[2/2] Semantic video search..."
        )

        video_output = (
            self.video_search.search(
                query
            )
        )

        video_results = (
            video_output.get(
                "reranked",
                []
            )
        )

        video_segments = (
            video_output.get(
                "segments",
                []
            )
        )

        # -------------------------------------------------
        # Unified fusion
        # -------------------------------------------------

        unified_results = (
            fuse_modalities(
                image_results=image_results,
                video_results=video_results,
                image_weight=IMAGE_WEIGHT,
                video_weight=VIDEO_WEIGHT
            )
        )

        elapsed_ms = (
            time.perf_counter()
            -
            started
        ) * 1000.0

        result = {

            "query_type":
                "text",

            "query":
                query,

            "fusion_method":
                "rank_weighted",

            "image_weight":
                IMAGE_WEIGHT,

            "video_weight":
                VIDEO_WEIGHT,

            "search_time_ms":
                round(
                    elapsed_ms,
                    3
                ),

            "image_result_count":
                len(
                    image_results
                ),

            "video_result_count":
                len(
                    video_results
                ),

            "video_segment_count":
                len(
                    video_segments
                ),

            "results":
                unified_results,

            "video_segments":
                video_segments
        }

        return result

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        result
    ):

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=4
            )

        print()
        print(
            f"[+] Saved: "
            f"{OUTPUT_FILE}"
        )


# =========================================================
# Main
# =========================================================

def main():

    searcher = (
        UnifiedSemanticSearch()
    )

    result = searcher.search(
        "빨간 패딩을 입은 남자"
    )

    print()
    print("=" * 70)
    print(
        "UNIFIED RESULTS"
    )
    print("=" * 70)

    for item in result[
        "results"
    ]:

        if (
            item["result_type"]
            == "image"
        ):

            print(
                f"[Rank "
                f"{item['unified_rank']}] "
                f"IMAGE"
            )

            print(
                f"  File       : "
                f"{item.get('filename')}"
            )

            print(
                f"  Similarity  : "
                f"{item.get('similarity', 0):.4f}"
            )

            print(
                f"  SourceRank : "
                f"{item['source_rank']}"
            )

            print(
                f"  Fusion     : "
                f"{item['fusion_score']:.4f}"
            )

        else:

            print(
                f"[Rank "
                f"{item['unified_rank']}] "
                f"VIDEO"
            )

            print(
                f"  Video      : "
                f"{item.get('video')}"
            )

            print(
                f"  Time       : "
                f"{item.get('time', 0):.3f}s"
            )

            print(
                f"  Reranker   : "
                f"{item.get('rerank_score', 0):.4f}"
            )

            print(
                f"  SourceRank : "
                f"{item['source_rank']}"
            )

            print(
                f"  Fusion     : "
                f"{item['fusion_score']:.4f}"
            )

    print()
    print("=" * 70)
    print(
        "VIDEO SEGMENTS"
    )
    print("=" * 70)

    for segment_rank, segment in enumerate(
        result["video_segments"],
        start=1
    ):

        video_name = segment.get(
            "video"
        )

        if not video_name:
            source = segment.get(
                "source",
                ""
            )

            video_name = (
                Path(source).name
                if source
                else "unknown"
            )

        print(
            f"[Segment {segment_rank}] "
            f"{video_name}"
        )

        print(
            f"  Time: "
            f"{segment['start_time']:.3f}"
            f" ~ "
            f"{segment['end_time']:.3f}"
        )

        print(
            f"  Best: "
            f"{segment['best_score']:.4f}"
        )

        print(
            f"  Avg : "
            f"{segment['average_score']:.4f}"
        )

    searcher.save(
        result
    )


if __name__ == "__main__":
    main()