import os
import json
import numpy as np
from pathlib import Path

from reid_model import PersonReID
from image_database import load_database


# =========================================================
# Existing Image Dataset
# =========================================================

CROP_DIR = "data/person_crops"
DATABASE_FILE = "data/images.json"

# =========================================================
# Market-1501 Dataset
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

MARKET_ROOT = (
    PROJECT_ROOT.parent
    / "datasets"
    / "Market-1501-v15.09.15"
)

MARKET_DIRS = [
    (
        MARKET_ROOT / "bounding_box_train",
        "train"
    ),
    (
        MARKET_ROOT / "bounding_box_test",
        "test"
    )
]

# =========================================================
# Output
# =========================================================

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

    # -----------------------------------------------------
    # Re-ID Model
    # -----------------------------------------------------

    model = PersonReID()

    embeddings = []
    metadata = []

    # =====================================================
    # 1. Existing forensic images
    # =====================================================

    images = load_database(
        DATABASE_FILE
    )

    image_count = 0

    if images:

        print()
        print(
            "=" * 60
        )
        print(
            "[*] EXISTING FORENSIC IMAGES"
        )
        print(
            "=" * 60
        )

        for image in images:

            for crop in image.get(
                "person_crops",
                []
            ):

                crop_path = crop.get(
                    "crop_path"
                )

                if (
                    not crop_path
                    or not os.path.exists(
                        crop_path
                    )
                ):

                    print(
                        f"[!] Crop not found, "
                        f"skipped: {crop_path}"
                    )

                    continue

                print(
                    f"[*] Processing: "
                    f"{crop_path}"
                )

                embedding = model.extract(
                    crop_path
                )

                embeddings.append(
                    embedding
                )

                metadata.append({

                    "source_type":
                        "image",

                    "source":
                        image["path"],

                    "filename":
                        image["filename"],

                    "image":
                        image["filename"],

                    "person_index":
                        crop[
                            "person_index"
                        ],

                    "bbox":
                        crop[
                            "bbox"
                        ],

                    "confidence":
                        crop[
                            "confidence"
                        ],

                    "detection_confidence":
                        crop[
                            "confidence"
                        ],

                    "crop":
                        crop[
                            "evidence_path"
                        ],

                    "evidence_path":
                        crop[
                            "evidence_path"
                        ]
                })

                image_count += 1

    else:

        print(
            "[!] Existing image database "
            "not found or empty."
        )

    # =====================================================
    # 2. Market-1501
    # =====================================================

    market_count = 0

    print()
    print(
        "=" * 60
    )
    print(
        "[*] MARKET-1501"
    )
    print(
        "=" * 60
    )

    for market_dir, split in MARKET_DIRS:

        if not market_dir.exists():

            print(
                f"[!] Market directory not found: "
                f"{market_dir}"
            )

            continue

        files = sorted(
            path
            for path in market_dir.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                }
            )
        )

        print()
        print(
            f"[*] Split: {split}"
        )

        print(
            f"[*] Files: {len(files)}"
        )

        for index, image_path in enumerate(
            files,
            start=1
        ):

            print(
                f"[{index:05d}/{len(files):05d}] "
                f"{image_path.name}"
            )

            embedding = model.extract(
                str(image_path)
            )

            embeddings.append(
                embedding
            )

            metadata.append({

                "source_type":
                    "market1501",

                "dataset":
                    "Market-1501",

                "split":
                    split,

                "source":
                    str(image_path),

                "filename":
                    image_path.name,

                "image":
                    image_path.name,

                # Market-1501 images are already
                # person bounding-box crops.
                "person_index":
                    0,

                "bbox":
                    None,

                "confidence":
                    None,

                "detection_confidence":
                    None,

                "crop":
                    str(image_path),

                "evidence_path":
                    str(image_path)
            })

            market_count += 1

    # =====================================================
    # 3. Validate
    # =====================================================

    if len(embeddings) == 0:

        print(
            "[!] 생성된 embedding이 없습니다."
        )

        return

    # =====================================================
    # 4. Save embeddings
    # =====================================================

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    np.save(
        EMBEDDING_FILE,
        embeddings
    )

    # =====================================================
    # 5. Save metadata
    # =====================================================

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

    # =====================================================
    # 6. Summary
    # =====================================================

    print()
    print(
        "=" * 60
    )
    print(
        "[*] Embedding 저장 완료"
    )
    print(
        "=" * 60
    )

    print(
        f"[*] Existing image entries : "
        f"{image_count}"
    )

    print(
        f"[*] Market-1501 entries    : "
        f"{market_count}"
    )

    print(
        f"[*] Total entries          : "
        f"{len(metadata)}"
    )

    print(
        f"[*] Embedding shape        : "
        f"{embeddings.shape}"
    )

    print(
        f"[*] 저장 위치              : "
        f"{EMBEDDING_FILE}"
    )

    print(
        f"[*] Metadata 위치          : "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()