import os
import json
import numpy as np

from reid_model import PersonReID
from similarity import cosine_similarity


EMBEDDING_FILE = (
    "data/embeddings/embeddings.npy"
)

METADATA_FILE = (
    "data/embeddings/metadata.json"
)

QUERY_IMAGE = (
    "data/person_crops/"
    "human1_person_0.jpg"
) #이미지 연관성 베이스 사진


def main():

    print("=" * 60)
    print("Person Re-ID Search")
    print("=" * 60)

    # 저장된 Embedding 불러오기
    embeddings = np.load(
        EMBEDDING_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        f"[*] Database embeddings: "
        f"{embeddings.shape}"
    )

    # OSNet
    model = PersonReID()

    print()
    print(
        f"[*] Query: {QUERY_IMAGE}"
    )

    # Query만 OSNet 실행
    query_embedding = model.extract(
        QUERY_IMAGE
    )

    print(
        f"[*] Query embedding: "
        f"{query_embedding.shape}"
    )

    results = []

    # 저장된 embedding과 비교
    for i, candidate_embedding in enumerate(
        embeddings
    ):

        candidate_path = metadata[i]["path"]

        # Query 자기 자신 제외
        if os.path.abspath(
            candidate_path
        ) == os.path.abspath(
            QUERY_IMAGE
        ):
            continue

        similarity = cosine_similarity(
            query_embedding,
            candidate_embedding
        )

        results.append({
            "filename":
                metadata[i]["filename"],

            "path":
                candidate_path,

            "similarity":
                similarity
        })

    # 높은 유사도 순
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    print()
    print("=" * 60)
    print("Search Results")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{result['filename']:<30} "
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )


if __name__ == "__main__":
    main()