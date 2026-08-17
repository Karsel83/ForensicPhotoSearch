from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys


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
# Paths
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-vl"
    / "Qwen3-VL-Embedding-2B"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "evidence"
    / "images"
    / "human1.jpg"
)


TEXT_QUERY = (
    "빨간 패딩을 입은 남자"
)


# =========================================================
# Similarity
# =========================================================

def cosine_similarity(
    vector_a,
    vector_b
):

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32
    ).reshape(-1)

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32
    ).reshape(-1)

    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )

    if (
        norm_a == 0
        or norm_b == 0
    ):
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        (
            norm_a
            * norm_b
        )
    )


# =========================================================
# Main
# =========================================================

def main():

    print(
        "[*] Loading Qwen3-VL Embedding..."
    )

    model = Qwen3VLEmbedder(
        model_name_or_path=
            str(MODEL_PATH)
    )

    print(
        "[+] Model loaded"
    )

    print()
    print(
        f"[*] Text query: "
        f"{TEXT_QUERY}"
    )

    # -----------------------------------------------------
    # Text embedding
    # -----------------------------------------------------

    text_inputs = [
        {
            "text":
                TEXT_QUERY,

            "instruction":
                "Retrieve images relevant to the user's query."
        }
    ]

    text_embedding = model.process(
        text_inputs
    )

    print(
        "[+] Text embedding shape:",
        text_embedding.shape
    )

    # -----------------------------------------------------
    # Image embedding
    # -----------------------------------------------------

    print()
    print(
        f"[*] Image: "
        f"{IMAGE_PATH}"
    )

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert(
        "RGB"
    )

    image_inputs = [
        {
            "image":
                image,

            "instruction":
                "Represent this image for image-text retrieval."
        }
    ]

    image_embedding = model.process(
        image_inputs
    )

    print(
        "[+] Image embedding shape:",
        image_embedding.shape
    )

    # -----------------------------------------------------
    # Convert
    # -----------------------------------------------------

    if torch.is_tensor(
        text_embedding
    ):

        text_vector = (
            text_embedding[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    else:

        text_vector = np.asarray(
            text_embedding[0]
        )

    if torch.is_tensor(
        image_embedding
    ):

        image_vector = (
            image_embedding[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    else:

        image_vector = np.asarray(
            image_embedding[0]
        )

    # -----------------------------------------------------
    # Similarity
    # -----------------------------------------------------

    score = cosine_similarity(
        text_vector,
        image_vector
    )

    print()
    print(
        "=" * 60
    )

    print(
        "TEXT → IMAGE TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"Text      : "
        f"{TEXT_QUERY}"
    )

    print(
        f"Image     : "
        f"{IMAGE_PATH.name}"
    )

    print(
        f"Text dim  : "
        f"{text_vector.shape[0]}"
    )

    print(
        f"Image dim : "
        f"{image_vector.shape[0]}"
    )

    print(
        f"Similarity: "
        f"{score:.6f}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()