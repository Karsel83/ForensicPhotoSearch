from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image


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
# Video Input
# =========================================================

VIDEO_DIR = (
    PROJECT_ROOT
    / "video"
    / "data"
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


# =========================================================
# Output
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "semantic_video"
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

SAMPLE_INTERVAL_SECONDS = 1.0

BATCH_SIZE = 4


# =========================================================
# Normalize
# =========================================================

def normalize(
    vectors
):

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
# Video Frame Sampling
# =========================================================

def sample_video_frames(
    video_path,
    interval_seconds
):

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Failed to open video: "
            f"{video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    current_time = 0.0

    samples = []

    while current_time <= duration:

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000.0
        )

        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        actual_frame = int(
            capture.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        ) - 1

        samples.append({

            "image":
                image,

            "frame":
                max(
                    actual_frame,
                    0
                ),

            "time":
                round(
                    current_time,
                    3
                ),

            "fps":
                fps
        })

        current_time += (
            interval_seconds
        )

    capture.release()

    return samples


# =========================================================
# Extract Batch Embeddings
# =========================================================

def extract_batch_embeddings(
    model,
    batch
):

    inputs = []

    for item in batch:

        inputs.append({

            "image":
                item["image"],

            "instruction":
                "Represent this video frame for image-text retrieval."
        })

    output = model.process(
        inputs
    )

    if torch.is_tensor(
        output
    ):

        return (
            output
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    return np.asarray(
        output,
        dtype=np.float32
    )


# =========================================================
# Main
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not VIDEO_DIR.exists():

        raise FileNotFoundError(
            f"Video directory not found: "
            f"{VIDEO_DIR}"
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
        "[+] Model loaded"
    )

    # -----------------------------------------------------
    # Video files
    # -----------------------------------------------------

    video_files = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    )

    print(
        f"[*] Videos: "
        f"{len(video_files)}"
    )

    if not video_files:

        print(
            "[!] No videos found."
        )

        return

    embeddings = []
    metadata = []

    # =====================================================
    # Process videos
    # =====================================================

    for video_index, video_path in enumerate(
        video_files,
        start=1
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"[Video {video_index}/{len(video_files)}]"
        )

        print(
            f"[*] {video_path.name}"
        )

        samples = sample_video_frames(
            video_path,
            SAMPLE_INTERVAL_SECONDS
        )

        print(
            f"[*] Sampled frames: "
            f"{len(samples)}"
        )

        # -------------------------------------------------
        # Batch processing
        # -------------------------------------------------

        for start in range(
            0,
            len(samples),
            BATCH_SIZE
        ):

            batch = samples[
                start:
                start + BATCH_SIZE
            ]

            print(
                f"[{start + 1:04d}/"
                f"{len(samples):04d}] "
                f"embedding..."
            )

            try:

                batch_embeddings = (
                    extract_batch_embeddings(
                        model,
                        batch
                    )
                )

            except Exception as error:

                print(
                    f"[!] Batch failed: "
                    f"{error}"
                )

                continue

            for item, vector in zip(
                batch,
                batch_embeddings
            ):

                embeddings.append(
                    vector
                )

                metadata.append({

                    "source_type":
                        "semantic_video",

                    "dataset":
                        "video",

                    "video":
                        video_path.name,

                    "source":
                        str(video_path),

                    "frame":
                        item["frame"],

                    "time":
                        item["time"],

                    "fps":
                        item["fps"],

                    "embedding_id":
                        len(
                            embeddings
                        ) - 1
                })

    # =====================================================
    # Validate
    # =====================================================

    if not embeddings:

        print(
            "[!] No video embeddings generated."
        )

        return

    embeddings = normalize(
        np.asarray(
            embeddings,
            dtype=np.float32
        )
    )

    # =====================================================
    # Save
    # =====================================================

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

    # =====================================================
    # Summary
    # =====================================================

    print()
    print(
        "=" * 60
    )

    print(
        "[*] Semantic video embedding complete"
    )

    print(
        f"[*] Videos: "
        f"{len(video_files)}"
    )

    print(
        f"[*] Frames: "
        f"{len(metadata)}"
    )

    print(
        f"[*] Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"[*] Embeddings: "
        f"{EMBEDDING_FILE}"
    )

    print(
        f"[*] Metadata: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()