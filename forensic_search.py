import os
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from reid_model import PersonReID
from similarity import cosine_similarity
from image_person_search import ImagePersonSearch
from person_detector import PersonDetector
from person_cropper import PersonCropper

from video.video_search import (
    VideoPersonSearch
)


# ============================================================
# Paths
# ============================================================

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "evidence",
    "images"
)

VIDEO_DIR = os.path.join(
    PROJECT_ROOT,
    "video",
    "data"
)

QUERY_IMAGE = os.path.join(
    VIDEO_DIR,
    "query.jpg"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "search_results.json"
)

IMAGE_EMBEDDINGS = os.path.join(
    PROJECT_ROOT,
    "data",
    "embeddings",
    "embeddings.npy"
)

IMAGE_METADATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "embeddings",
    "metadata.json"
)


# ============================================================
# Time
# ============================================================

def format_time(seconds):

    seconds = float(seconds)

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds
        - minutes * 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:06.3f}"
    )


# ============================================================
# Image Search
# ============================================================

# ============================================================
# Image Search
# ============================================================

def search_images(
    query_feature,
    model
):

    print()
    print(
        "=" * 60
    )

    print(
        "[*] IMAGE SEARCH"
    )

    print(
        "=" * 60
    )

    if not os.path.exists(
        IMAGE_DIR
    ):

        print(
            f"[!] Image directory not found: "
            f"{IMAGE_DIR}"
        )

        return []

    # --------------------------------------------------------
    # Image Search Engine
    # --------------------------------------------------------

    searcher = ImagePersonSearch(
        reid=model,
        embeddings_path=IMAGE_EMBEDDINGS,
        metadata_path=IMAGE_METADATA,
        top_k=50
    )

    # --------------------------------------------------------
    # Evidence directory
    # --------------------------------------------------------

    evidence_dir = os.path.join(
        PROJECT_ROOT,
        "evidence",
        "images"
    )

    os.makedirs(
        evidence_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = searcher.search_directory(

        query_feature=
            query_feature,

        image_dir=
            IMAGE_DIR,

        evidence_dir=
            evidence_dir
    )

    # --------------------------------------------------------
    # Result metadata
    # --------------------------------------------------------

    for result in results:

        result[
            "source_type"
        ] = "image"

    print()
    print(
        f"[*] Image search results: "
        f"{len(results)}"
    )

    return results


# ============================================================
# Video Search
# ============================================================

def search_videos(
    query_feature,
    model
):

    print()
    print(
        "=" * 60
    )

    print(
        "[*] VIDEO SEARCH"
    )

    print(
        "=" * 60
    )

    if not os.path.exists(
        VIDEO_DIR
    ):

        print(
            f"[!] 영상 폴더 없음: "
            f"{VIDEO_DIR}"
        )

        return []

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv"
    }

    video_files = []

    for path in Path(
        VIDEO_DIR
    ).iterdir():

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            in video_extensions
        ):

            video_files.append(
                str(path)
            )

    video_files.sort()

    print(
        f"[*] 영상 파일: "
        f"{len(video_files)}개"
    )

    if not video_files:

        return []

    searcher = VideoPersonSearch(
        reid=model
    )

    all_results = []

    for video_path in video_files:

        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        video_evidence_dir = os.path.join(
            PROJECT_ROOT,
            "video",
            "evidence",
            video_name
        )

        searcher = VideoPersonSearch(
            reid=model,
            evidence_root=video_evidence_dir
        )

        results = searcher.search(

            query_feature=
                query_feature,

            video_path=
                video_path,

            threshold=0.80,

            high_score_threshold=0.75,

            min_high_score_count=5,

            reid_interval=10
        )

        for result in results:

            result[
                "source_type"
            ] = "video"

            result[
                "source"
            ] = video_path

            result[
                "video"
            ] = os.path.basename(
                video_path
            )

            result[
                "timecode"
            ] = format_time(
                result[
                    "best_time"
                ]
            )

            all_results.append(
                result
            )

    all_results.sort(
        key=lambda x:
            x["best_score"],
        reverse=True
    )

    return all_results


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FORENSIC UNIFIED SEARCH")
    print("=" * 70)

    # ========================================================
    # Query 확인
    # ========================================================

    print()
    print(f"[*] Query: {QUERY_IMAGE}")

    if not os.path.exists(QUERY_IMAGE):

        print(
            "[ERROR] Query 이미지가 없습니다."
        )

        print(
            f"       {QUERY_IMAGE}"
        )

        return

    # ========================================================
    # Re-ID Model
    # ========================================================

    print()
    print("[*] Loading OSNet...")

    model = PersonReID()

    # ========================================================
    # Query Feature
    # ========================================================

    print()
    print("[*] Extracting query feature...")

    query_feature = model.extract(
        QUERY_IMAGE
    )

    print(
        f"[*] Query feature shape: "
        f"{query_feature.shape}"
    )

    # ========================================================
    # Image Search
    # ========================================================

    image_results = search_images(
        query_feature,
        model
    )

    print(
        f"[*] Image results: "
        f"{len(image_results)}"
    )

    # ========================================================
    # Video Search
    # ========================================================

    video_results = search_videos(
        query_feature,
        model
    )

    print(
        f"[*] Video results: "
        f"{len(video_results)}"
    )

    # ========================================================
    # 통합 결과
    # ========================================================

    merged = []

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    for result in image_results:

        merged.append({

            "source_type":
                "image",
            "result_id":
                f"{result['image']}",

            "similarity":
                result["similarity"],

            "source":
                result["source"],

            "filename":
                result["filename"],

            "crop":
                result["crop"],

            "bbox":
                result["bbox"],

            "confidence":
                result["confidence"],

            "track_id":
                None,

            "best_frame":
                None,

            "best_time":
                None,

            "timecode":
                None,

            "first_time":
                None,

            "last_time":
                None,

            "duration":
                None,

            "evidence_path":
                result["crop"],

            "metadata_path":
                None
        })

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    for result in video_results:

        merged.append({
            
            "source_type":
                "video",

            "result_id":
                f"{result['video']}:track_{result['track_id']}",

           "similarity":
                result["best_score"],

            "best_score":
                result["best_score"],

            "average_score":
                result.get(
                    "average_score"
                ),

            "high_score_count":
                result.get(
                    "high_score_count"
                ),

            "sample_count":
                result.get(
                    "sample_count"
                ),

            "high_score_ratio":
                result.get(
                    "high_score_ratio"
                ),

            "track_score":
                result.get(
                    "track_score"
                ),
            "match_segments":
                result.get(
                    "match_segments",
                    []
                ),
            "source":
                result["source"],
            "evidence":
                result.get(
                    "evidence",
                    {}
                ),

            "video":
                result["video"],

            "track_id":
                result["track_id"],

            "match":
                result["match"],

            "best_frame":
                result["best_frame"],

            "best_time":
                result["best_time"],

            "timecode":
                result["timecode"],

            "first_time":
                result["first_time"],

            "last_time":
                result["last_time"],

            "duration":
                result["duration"],

            "evidence_path":
                result.get(
                    "evidence_path"
                ),

            "metadata_path":
                result.get(
                    "metadata_path"
                )
        })

    # ========================================================
    # Ranking
    # ========================================================

    merged.sort(
        key=lambda x:
            x["similarity"],
        reverse=True
    )

    for rank, result in enumerate(
        merged,
        start=1
    ):

        result["rank"] = rank

    # ========================================================
    # JSON 저장
    # ========================================================

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )
    video_summary = {}

    for result in video_results:

        video_name = result["video"]

        if video_name not in video_summary:

            video_summary[video_name] = {
                "track_count": 0,
                "best_score": 0.0,
                "best_track_id": None,
                "match_count": 0
            }

        summary = video_summary[video_name]

        summary["track_count"] += 1

        if result["best_score"] > summary["best_score"]:

            summary["best_score"] = result["best_score"]

            summary["best_track_id"] = (
                result["track_id"]
            )

        if result["match"]:

            summary["match_count"] += 1
    output = {

        "query": {
            "path":
                QUERY_IMAGE
        },
        "video_count": len(video_summary),

        "videos": video_summary,

        "image_result_count":
            len(image_results),

        "video_result_count":
            len(video_results),

        "total_result_count":
            len(merged),

        "results":
            merged
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 70)
    print("INTEGRATED SEARCH RESULTS")
    print("=" * 70)

    for result in merged[:20]:

        print()

        print(
            f"[Rank {result['rank']}] "
            f"{result['source_type'].upper()}"
        )
        print(
            f"  Result ID  : "
            f"{result['result_id']}"
        )
        print(
            f"  Similarity : "
            f"{result['similarity']:.4f}"
        )
        if result["source_type"] == "video":

            print(
                f"  TrackScore : "
                f"{result.get('track_score', 0):.4f}"
            )

            print(
                f"  Avg Score  : "
                f"{result.get('average_score', 0):.4f}"
            )

            print(
                f"  High Ratio : "
                f"{result.get('high_score_ratio', 0):.4f}"
            )
        if result["source_type"] == "image":

            print(
                f"  Image      : "
                f"{result['filename']}"
            )

        else:

            print(
                f"  Video      : "
                f"{result['video']}"
            )

            print(
                f"  Track      : "
                f"{result['track_id']}"
            )

            print(
                f"  Time       : "
                f"{result['timecode']}"
            )

            print(
                f"  Duration   : "
                f"{result['duration']:.3f}s"
            )

            segments = result.get(
                "match_segments",
                []
            )

            if segments:

                print(
                    "  Match Segments:"
                )

                for segment in segments:

                    print(
                        f"    "
                        f"{format_time(segment['start_time'])}"
                        f" ~ "
                        f"{format_time(segment['end_time'])}"
                        f" "
                        f"(best="
                        f"{segment['best_score']:.4f})"
                    )

    if segments:

        print(
            "  Match Segments:"
        )

        for segment in segments:

            print(
                f"    "
                f"{format_time(segment['start_time'])}"
                f" ~ "
                f"{format_time(segment['end_time'])}"
                f" "
                f"(best="
                f"{segment['best_score']:.4f})"
            )

    print()
    print("=" * 70)

    print(
        f"[INFO] 결과 저장: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
