from pathlib import Path

from qwen_video_reranker import (
    QwenVideoReranker
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "evidence"
    / "images"
    / "human1.jpg"
)


def main():

    query = (
        "빨간 패딩을 입은 남자"
    )

    reranker = QwenVideoReranker()

    score = reranker.score_image(
        query,
        IMAGE_PATH
    )

    print()
    print("=" * 60)
    print("QWEN RERANKER TEST")
    print("=" * 60)

    print(
        f"Query : {query}"
    )

    print(
        f"Image : {IMAGE_PATH}"
    )

    print(
        f"Score : {score:.6f}"
    )


if __name__ == "__main__":
    main()