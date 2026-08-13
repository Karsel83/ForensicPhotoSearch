import os
import json
import numpy as np

from reid_model import PersonReID


CROP_DIR = "data/person_crops"
OUTPUT_DIR = "data/embeddings"

EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "embeddings.npy"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    model = PersonReID()

    embeddings = []
    metadata = []

    files = sorted(os.listdir(CROP_DIR))

    for filename in files:

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            CROP_DIR,
            filename
        )

        print(f"[*] Processing: {filename}")

        embedding = model.extract(
            image_path
        )

        embeddings.append(embedding)

        metadata.append({
            "filename": filename,
            "path": image_path
        })

    if len(embeddings) == 0:

        print("[!] 이미지가 없습니다.")
        return

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    np.save(
        EMBEDDING_FILE,
        embeddings
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print("=" * 60)
    print("[*] Embedding 저장 완료")
    print("=" * 60)

    print(
        f"[*] 이미지 수: {len(metadata)}"
    )

    print(
        f"[*] Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"[*] 저장 위치: "
        f"{EMBEDDING_FILE}"
    )


if __name__ == "__main__":
    main()