from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
import torch


# =========================================================
# Project Root
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

QWEN_ROOT = (
    PROJECT_ROOT
    / "Qwen3-VL-Embedding"
)

if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(QWEN_ROOT)
    )

from src.models.qwen3_vl_embedding import (
    Qwen3VLEmbedder
)


# =========================================================
# Model
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-vl"
    / "Qwen3-VL-Embedding-2B"
)


# =========================================================
# Existing forensic images
# =========================================================

FORENSIC_IMAGE_DIR = (
    PROJECT_ROOT
    / "evidence"
    / "images"
)


# =========================================================
# Market-1501
#
# Actual path:
# C:\Users\Karsel\Desktop\forensic tool\datasets\
# Market-1501-v15.09.15
# =========================================================

MARKET_ROOT = (
    PROJECT_ROOT.parent
    / "datasets"
    / "Market-1501-v15.09.15"
)

MARKET_DIRS = [
    (
        MARKET_ROOT
        / "bounding_box_train",
        "train"
    ),
    (
        MARKET_ROOT
        / "bounding_box_test",
        "test"
    )
]


# =========================================================
# Output
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "semantic"
)

EMBEDDING_FILE = (
    OUTPUT_DIR
    / "embeddings.npy"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "metadata.json"
)


# =========================================================
# Settings
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ---------------------------------------------------------
# First test:
# 500 Market images per split
#
# For the full dataset later:
# MAX_MARKET_PER_SPLIT = None
# ---------------------------------------------------------

MAX_MARKET_PER_SPLIT = None


# =========================================================
# Normalize
# =========================================================

def normalize(vectors):

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
# Extract embedding
# =========================================================

def extract_image_embedding(
    model,
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    inputs = [
        {
            "image":
                image,

            "instruction":
                "Represent this image for image-text retrieval."
        }
    ]

    embedding = model.process(
        inputs
    )

    if torch.is_tensor(
        embedding
    ):

        vector = (
            embedding[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    else:

        vector = np.asarray(
            embedding[0],
            dtype=np.float32
        )

    return vector


# =========================================================
# Main
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print(
        "[*] Loading Qwen3-VL "
        "Embedding model..."
    )

    model = Qwen3VLEmbedder(
        model_name_or_path=
            str(MODEL_PATH)
    )

    print(
        "[+] Qwen3-VL "
        "Embedding loaded"
    )

    embeddings = []
    metadata = []

    # =====================================================
    # 1. Existing forensic images
    # =====================================================

    print()
    print(
        "=" * 60
    )

    print(
        "[*] FORENSIC IMAGES"
    )

    print(
        "=" * 60
    )

    forensic_files = []

    if FORENSIC_IMAGE_DIR.exists():

        forensic_files = sorted(
            path
            for path
            in FORENSIC_IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

    print(
        f"[*] Forensic images: "
        f"{len(forensic_files)}"
    )

    for index, image_path in enumerate(
        forensic_files,
        start=1
    ):

        print(
            f"[F {index:04d}/{len(forensic_files):04d}] "
            f"{image_path.name}"
        )

        try:

            vector = extract_image_embedding(
                model,
                image_path
            )

            embeddings.append(
                vector
            )

            metadata.append({

                "source_type":
                    "forensic_image",

                "dataset":
                    "forensic",

                "split":
                    None,

                "source":
                    str(image_path),

                "filename":
                    image_path.name,

                "image":
                    image_path.name,

                "embedding_id":
                    len(embeddings) - 1
            })

        except Exception as error:

            print(
                f"[!] Failed: "
                f"{image_path.name}"
            )

            print(
                f"    {error}"
            )

    # =====================================================
    # 2. Market-1501
    # =====================================================

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

    market_total = 0

    for market_dir, split in MARKET_DIRS:

        if not market_dir.exists():

            print(
                f"[!] Market directory not found: "
                f"{market_dir}"
            )

            continue

        files = sorted(
            path
            for path
            in market_dir.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        original_count = len(
            files
        )

        if (
            MAX_MARKET_PER_SPLIT
            is not None
        ):

            files = files[
                :MAX_MARKET_PER_SPLIT
            ]

        print()
        print(
            f"[*] Split: {split}"
        )

        print(
            f"[*] Available: "
            f"{original_count}"
        )

        print(
            f"[*] Processing: "
            f"{len(files)}"
        )

        for index, image_path in enumerate(
            files,
            start=1
        ):

            print(
                f"[M-{split} "
                f"{index:04d}/{len(files):04d}] "
                f"{image_path.name}"
            )

            try:

                vector = extract_image_embedding(
                    model,
                    image_path
                )

                embeddings.append(
                    vector
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

                    "embedding_id":
                        len(embeddings) - 1
                })

                market_total += 1

            except Exception as error:

                print(
                    f"[!] Failed: "
                    f"{image_path.name}"
                )

                print(
                    f"    {error}"
                )

    # =====================================================
    # 3. Validate
    # =====================================================

    if not embeddings:

        print(
            "[!] No embeddings generated."
        )

        return

    embeddings = normalize(
        np.asarray(
            embeddings,
            dtype=np.float32
        )
    )

    # =====================================================
    # 4. Save embeddings
    # =====================================================

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

    forensic_count = sum(
        1
        for item in metadata
        if item["source_type"]
        == "forensic_image"
    )

    market_count = sum(
        1
        for item in metadata
        if item["source_type"]
        == "market1501"
    )

    train_count = sum(
        1
        for item in metadata
        if (
            item["source_type"]
            == "market1501"
            and
            item["split"]
            == "train"
        )
    )

    test_count = sum(
        1
        for item in metadata
        if (
            item["source_type"]
            == "market1501"
            and
            item["split"]
            == "test"
        )
    )

    print()
    print(
        "=" * 60
    )

    print(
        "[*] Semantic embedding build complete"
    )

    print(
        f"[*] Forensic images : "
        f"{forensic_count}"
    )

    print(
        f"[*] Market total    : "
        f"{market_count}"
    )

    print(
        f"[*] Market train    : "
        f"{train_count}"
    )

    print(
        f"[*] Market test     : "
        f"{test_count}"
    )

    print(
        f"[*] Total           : "
        f"{len(metadata)}"
    )

    print(
        f"[*] Embedding shape : "
        f"{embeddings.shape}"
    )

    print(
        f"[*] Embeddings      : "
        f"{EMBEDDING_FILE}"
    )

    print(
        f"[*] Metadata        : "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()