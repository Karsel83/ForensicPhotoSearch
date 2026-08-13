import os
import json

from video_search import VideoPersonSearch


QUERY_IMAGE = "data/query.jpg"
VIDEO_FILE = "data/test.mp4"

EVIDENCE_DIR = "evidence"

RESULT_FILE = "results/video_results.json"


def main():

    print("=" * 60)
    print("FORENSIC VIDEO PERSON SEARCH")
    print("=" * 60)

    # Query 이미지 확인
    if not os.path.exists(QUERY_IMAGE):
        print(
            f"[ERROR] Query 이미지가 없습니다: "
            f"{QUERY_IMAGE}"
        )
        return

    # 영상 확인
    if not os.path.exists(VIDEO_FILE):
        print(
            f"[ERROR] 영상이 없습니다: "
            f"{VIDEO_FILE}"
        )
        return

    # Video Search 생성
    searcher = VideoPersonSearch()

    # 영상 분석
    results = searcher.search(
        query_path=QUERY_IMAGE,
        video_path=VIDEO_FILE,
        evidence_dir=EVIDENCE_DIR,
        threshold=0.80,
        high_score_threshold=0.75,
        min_high_score_count=5,
        reid_interval=10
    )

    # 결과 디렉터리 생성
    os.makedirs(
        "results",
        exist_ok=True
    )

    # JSON 저장
    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # 결과 출력
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    for result in results:

        print(
            f"\nTrack #{result['track_id']}"
        )

        print(
            f"  Match      : "
            f"{result['match']}"
        )

        print(
            f"  Best Score : "
            f"{result['best_score']}"
        )

        print(
            f"  Average    : "
            f"{result['average_score']}"
        )

        print(
            f"  Time       : "
            f"{result['first_time']} "
            f"~ "
            f"{result['last_time']}"
        )

        print(
            f"  Evidence   : "
            f"{result['evidence_path']}"
        )

    print()
    print(
        f"[INFO] 결과 저장: {RESULT_FILE}"
    )


if __name__ == "__main__":
    main()