import os
import json
import numpy as np

from reid_model import PersonReID
from image_database import load_database


CROP_DIR = "data/person_crops"
DATABASE_FILE = "data/images.json"
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

    images = load_database(DATABASE_FILE)

    if not images:
        print("[!] Image database not found or empty. Run build_database.py first.")
        return

    for image in images:
        for crop in image.get("person_crops", []):
            crop_path = crop.get("crop_path")

            if not crop_path or not os.path.exists(crop_path):
                print(f"[!] Crop not found, skipped: {crop_path}")
                continue

            print(f"[*] Processing: {crop_path}")
            embeddings.append(model.extract(crop_path))

            # This is the result schema required by forensic_search.py.  The
            # vector index therefore needs no image decoding at search time.
            metadata.append({
                "source_type": "image",
                "source": image["path"],
                "filename": image["filename"],
                "image": image["filename"],
                "person_index": crop["person_index"],
                "bbox": crop["bbox"],
                "confidence": crop["confidence"],
                "detection_confidence": crop["confidence"],
                "crop": crop["evidence_path"],
                "evidence_path": crop["evidence_path"]
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
