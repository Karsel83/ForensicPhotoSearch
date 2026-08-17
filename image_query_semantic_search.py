from pathlib import Path
import json
import os
import sys

import cv2
import faiss
import numpy as np
import torch
from PIL import Image

from qwen_video_reranker import QwenVideoReranker


# =========================================================
# Project
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

SEMANTIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "semantic"
)

SEMANTIC_VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "semantic_video"
)

FAISS_ROOT = Path(
    os.environ.get(
        "FORENSIC_FAISS_ROOT",
        r"C:\ForensicFAISS"
    )
)

PROJECT_IMAGE_INDEX = (
    SEMANTIC_DIR
    / "image.index"
)

FALLBACK_IMAGE_INDEX = (
    FAISS_ROOT
    / "semantic"
    / "image.index"
)

IMAGE_INDEX = (
    PROJECT_IMAGE_INDEX
    if PROJECT_IMAGE_INDEX.exists()
    else FALLBACK_IMAGE_INDEX
)

IMAGE_METADATA = (
    SEMANTIC_DIR
    / "metadata.json"
)

PROJECT_VIDEO_INDEX = (
    SEMANTIC_VIDEO_DIR
    / "video.index"
)

FALLBACK_VIDEO_INDEX = (
    FAISS_ROOT
    / "semantic_video"
    / "video.index"
)

VIDEO_INDEX = (
    PROJECT_VIDEO_INDEX
    if PROJECT_VIDEO_INDEX.exists()
    else FALLBACK_VIDEO_INDEX
)

VIDEO_METADATA = (
    SEMANTIC_VIDEO_DIR
    / "metadata.json"
)

TEMP_FRAME_DIR = (
    FAISS_ROOT
    / "image_query_frames"
)


# =========================================================
# Settings
# =========================================================

IMAGE_CANDIDATE_K = 20
VIDEO_CANDIDATE_K = 20

FINAL_IMAGE_K = 10
FINAL_VIDEO_K = 10


# =========================================================
# Helpers
# =========================================================

def normalize_vector(
    vector
):

    vector = np.asarray(
        vector,
        dtype=np.float32
    ).reshape(-1)

    norm = np.linalg.norm(
        vector
    )

    if norm > 0:
        vector /= norm

    return vector


def load_json(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def tensor_to_float(
    value
):

    if torch.is_tensor(
        value
    ):

        return float(
            value
            .detach()
            .float()
            .cpu()
            .item()
        )

    return float(
        value
    )


# =========================================================
# Image Query Search
# =========================================================

class ImageQuerySemanticSearch:

    def __init__(
        self
    ):

        # -------------------------------------------------
        # Embedding model
        # -------------------------------------------------

        print(
            "[*] Loading Qwen3-VL "
            "Embedding..."
        )

        self.embedding_model = (
            Qwen3VLEmbedder(
                model_name_or_path=
                    str(MODEL_PATH)
            )
        )

        print(
            "[+] Embedding model loaded"
        )

        # -------------------------------------------------
        # Image FAISS
        # -------------------------------------------------

        print(
            "[*] Loading image FAISS..."
        )

        if not IMAGE_INDEX.exists():
            raise FileNotFoundError(
                f"Image FAISS index not found:\n"
                f"{IMAGE_INDEX}"
            )

        if not IMAGE_METADATA.exists():
            raise FileNotFoundError(
                f"Image metadata not found:\n"
                f"{IMAGE_METADATA}"
            )

        self.image_index = (
            faiss.read_index(
                str(IMAGE_INDEX)
            )
        )

        self.image_metadata = (
            load_json(
                IMAGE_METADATA
            )
        )

        if (
            self.image_index.ntotal
            != len(self.image_metadata)
        ):
            raise ValueError(
                "Image FAISS index and metadata "
                "length do not match."
            )

        print(
            f"[+] Image vectors: "
            f"{self.image_index.ntotal}"
        )

        # -------------------------------------------------
        # Video FAISS
        # -------------------------------------------------

        print(
            "[*] Loading video FAISS..."
        )

        if not VIDEO_INDEX.exists():
            raise FileNotFoundError(
                f"Video FAISS index not found:\n"
                f"{VIDEO_INDEX}"
            )

        if not VIDEO_METADATA.exists():
            raise FileNotFoundError(
                f"Video metadata not found:\n"
                f"{VIDEO_METADATA}"
            )

        self.video_index = (
            faiss.read_index(
                str(VIDEO_INDEX)
            )
        )

        self.video_metadata = (
            load_json(
                VIDEO_METADATA
            )
        )

        if (
            self.video_index.ntotal
            != len(self.video_metadata)
        ):
            raise ValueError(
                "Video FAISS index and metadata "
                "length do not match."
            )

        print(
            f"[+] Video vectors: "
            f"{self.video_index.ntotal}"
        )

        # -------------------------------------------------
        # Reranker
        # -------------------------------------------------

        print(
            "[*] Loading Qwen3-VL "
            "Reranker..."
        )

        self.reranker = (
            QwenVideoReranker()
        )

        print(
            "[+] Unified image-query "
            "search ready"
        )

        TEMP_FRAME_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # Query Image Embedding
    # =====================================================

    def encode_image(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Query image not found:\n"
                f"{image_path}"
            )

        image = (
            Image.open(
                image_path
            )
            .convert("RGB")
        )

        inputs = [
            {
                "image":
                    image,

                "instruction":
                    (
                        "Represent this image "
                        "for image-text retrieval."
                    )
            }
        ]

        embedding = (
            self.embedding_model.process(
                inputs
            )
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

        return normalize_vector(
            vector
        )

    # =====================================================
    # Stage 1: Image FAISS
    # =====================================================

    def search_images(
        self,
        query_vector
    ):

        query_vector = (
            query_vector
            .reshape(1, -1)
            .astype(np.float32)
        )

        candidate_k = min(
            IMAGE_CANDIDATE_K,
            self.image_index.ntotal
        )

        scores, ids = (
            self.image_index.search(
                query_vector,
                candidate_k
            )
        )

        results = []

        for rank, (
            score,
            vector_id
        ) in enumerate(
            zip(
                scores[0],
                ids[0]
            ),
            start=1
        ):

            if vector_id < 0:
                continue

            item = dict(
                self.image_metadata[
                    int(vector_id)
                ]
            )

            item.update({

                "candidate_rank":
                    rank,

                "faiss_score":
                    float(score),

                "vector_id":
                    int(vector_id)
            })

            results.append(
                item
            )

        return results

    # =====================================================
    # Stage 1: Video FAISS
    # =====================================================

    def search_videos(
        self,
        query_vector
    ):

        query_vector = (
            query_vector
            .reshape(1, -1)
            .astype(np.float32)
        )

        candidate_k = min(
            VIDEO_CANDIDATE_K,
            self.video_index.ntotal
        )

        scores, ids = (
            self.video_index.search(
                query_vector,
                candidate_k
            )
        )

        results = []

        for rank, (
            score,
            vector_id
        ) in enumerate(
            zip(
                scores[0],
                ids[0]
            ),
            start=1
        ):

            if vector_id < 0:
                continue

            item = dict(
                self.video_metadata[
                    int(vector_id)
                ]
            )

            item.update({

                "candidate_rank":
                    rank,

                "faiss_score":
                    float(score),

                "vector_id":
                    int(vector_id)
            })

            results.append(
                item
            )

        return results

    # =====================================================
    # Stage 2: Image → Image
    # =====================================================

    def rerank_images(
        self,
        query_path,
        candidates
    ):

        results = []

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                f"[Image rerank "
                f"{index:02d}/"
                f"{len(candidates):02d}] "
                f"{candidate['filename']}"
            )

            try:

                candidate_source = (
                    candidate.get(
                        "source"
                    )
                )

                if not candidate_source:
                    print(
                        "[!] Missing image source."
                    )
                    continue

                score = (
                    self.reranker
                    .score_image_to_image(
                        query_path,
                        candidate_source
                    )
                )

                item = dict(
                    candidate
                )

                item["rerank_score"] = float(
                    score
                )

                results.append(
                    item
                )

            except Exception as error:

                print(
                    f"[!] Image rerank failed: "
                    f"{error}"
                )

        results.sort(
            key=lambda item:
                item["rerank_score"],
            reverse=True
        )

        return results[
            :FINAL_IMAGE_K
        ]

    # =====================================================
    # Extract Video Frame
    # =====================================================

    def extract_video_frame(
        self,
        video_source,
        frame_number,
        output_path
    ):

        capture = cv2.VideoCapture(
            str(video_source)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open video:\n"
                f"{video_source}"
            )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )

        success, frame = (
            capture.read()
        )

        capture.release()

        if not success:
            raise RuntimeError(
                f"Failed to extract frame "
                f"{frame_number} from:\n"
                f"{video_source}"
            )

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        if not success:
            raise RuntimeError(
                f"Failed to save frame:\n"
                f"{output_path}"
            )

        return output_path

    # =====================================================
    # Stage 2: Image → Video Frame
    # =====================================================

    def rerank_videos(
        self,
        query_path,
        candidates
    ):

        results = []

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            video_name = (
                candidate.get(
                    "video",
                    Path(
                        candidate["source"]
                    ).name
                )
            )

            frame_number = int(
                candidate["frame"]
            )

            print(
                f"[Video rerank "
                f"{index:02d}/"
                f"{len(candidates):02d}] "
                f"{video_name} "
                f"frame={frame_number}"
            )

            try:

                video_source = Path(
                    candidate["source"]
                )

                if not video_source.exists():
                    video_source = (
                        PROJECT_ROOT
                        / "video"
                        / "data"
                        / Path(
                            candidate["video"]
                        ).name
                    )

                if not video_source.exists():
                    raise FileNotFoundError(
                        f"Video not found: {video_source}"
                    )

                # -----------------------------------------
                # Extract candidate frame
                # -----------------------------------------

                TEMP_FRAME_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                temp_path = (
                    TEMP_FRAME_DIR
                    / (
                        f"{video_source.stem}"
                        f"_frame_{frame_number}.jpg"
                    )
                )

                self.extract_video_frame(
                    video_source,
                    frame_number,
                    temp_path
                )

                # -----------------------------------------
                # Image → Video Frame Reranker
                # -----------------------------------------

                score = (
                    self.reranker
                    .score_image_to_video(
                        query_path,
                        temp_path
                    )
                )

                item = dict(
                    candidate
                )

                item.update({

                    "rerank_score":
                        float(score),

                    "rerank_frame":
                        str(temp_path)
                })

                results.append(
                    item
                )

            except Exception as error:

                print(
                    f"[!] Video rerank failed: "
                    f"{error}"
                )

        results.sort(
            key=lambda item:
                item["rerank_score"],
            reverse=True
        )

        return results[
            :FINAL_VIDEO_K
        ]

    # =====================================================
    # Video Segment Grouping
    # =====================================================

    def group_video_segments(
        self,
        video_results,
        max_gap_seconds=1.0
    ):

        grouped = {}

        for result in video_results:

            video_name = result.get(
                "video"
            )

            if not video_name:
                continue

            grouped.setdefault(
                video_name,
                []
            ).append(
                result
            )

        segments = []

        for video_name, frames in (
            grouped.items()
        ):

            frames.sort(
                key=lambda item:
                    float(
                        item.get(
                            "time",
                            0.0
                        )
                    )
            )

            current = None

            for frame in frames:

                time_value = float(
                    frame.get(
                        "time",
                        0.0
                    )
                )

                score = float(
                    frame.get(
                        "rerank_score",
                        0.0
                    )
                )

                if current is None:

                    current = {

                        "video":
                            video_name,

                        "source":
                            frame.get(
                                "source"
                            ),

                        "start_time":
                            time_value,

                        "end_time":
                            time_value,

                        "best_time":
                            time_value,

                        "best_frame":
                            frame.get(
                                "frame"
                            ),

                        "best_score":
                            score,

                        "frames":
                            [
                                frame
                            ]
                    }

                    continue

                gap = (
                    time_value
                    -
                    current["end_time"]
                )

                if gap <= max_gap_seconds:

                    current["end_time"] = (
                        time_value
                    )

                    current["frames"].append(
                        frame
                    )

                    if (
                        score
                        >
                        current["best_score"]
                    ):

                        current["best_score"] = (
                            score
                        )

                        current["best_time"] = (
                            time_value
                        )

                        current["best_frame"] = (
                            frame.get(
                                "frame"
                            )
                        )

                else:

                    current["duration"] = round(
                        current["end_time"]
                        -
                        current["start_time"],
                        3
                    )

                    current["average_score"] = (
                        sum(
                            item[
                                "rerank_score"
                            ]
                            for item in
                            current["frames"]
                        )
                        /
                        len(
                            current["frames"]
                        )
                    )

                    segments.append(
                        current
                    )

                    current = {

                        "video":
                            video_name,

                        "source":
                            frame.get(
                                "source"
                            ),

                        "start_time":
                            time_value,

                        "end_time":
                            time_value,

                        "best_time":
                            time_value,

                        "best_frame":
                            frame.get(
                                "frame"
                            ),

                        "best_score":
                            score,

                        "frames":
                            [
                                frame
                            ]
                    }

            if current is not None:

                current["duration"] = round(
                    current["end_time"]
                    -
                    current["start_time"],
                    3
                )

                current["average_score"] = (
                    sum(
                        item[
                            "rerank_score"
                        ]
                        for item in
                        current["frames"]
                    )
                    /
                    len(
                        current["frames"]
                    )
                )

                segments.append(
                    current
                )

        segments.sort(
            key=lambda item:
                item["best_score"],
            reverse=True
        )

        for rank, segment in enumerate(
            segments,
            start=1
        ):

            segment["segment_rank"] = (
                rank
            )

        return segments

    # =====================================================
    # Full Search
    # =====================================================

    def search(
        self,
        query_image
    ):

        query_image = Path(
            query_image
        )

        if not query_image.exists():

            raise FileNotFoundError(
                f"Query image not found:\n"
                f"{query_image}"
            )

        print()
        print("=" * 70)
        print(
            "IMAGE → IMAGE + VIDEO SEARCH"
        )
        print("=" * 70)

        print(
            f"[*] Query image: "
            f"{query_image}"
        )

        # -------------------------------------------------
        # Query embedding
        # -------------------------------------------------

        query_vector = (
            self.encode_image(
                query_image
            )
        )

        # -------------------------------------------------
        # Stage 1
        # -------------------------------------------------

        print()
        print(
            "[*] Stage 1: Image FAISS"
        )

        image_candidates = (
            self.search_images(
                query_vector
            )
        )

        print(
            f"[+] Image candidates: "
            f"{len(image_candidates)}"
        )

        print()
        print(
            "[*] Stage 1: Video FAISS"
        )

        video_candidates = (
            self.search_videos(
                query_vector
            )
        )

        print(
            f"[+] Video candidates: "
            f"{len(video_candidates)}"
        )

        # -------------------------------------------------
        # Stage 2
        # -------------------------------------------------

        print()
        print(
            "[*] Stage 2: Image → Image Reranking"
        )

        image_results = (
            self.rerank_images(
                query_image,
                image_candidates
            )
        )

        print()
        print(
            "[*] Stage 2: Image → Video Reranking"
        )

        video_results = (
            self.rerank_videos(
                query_image,
                video_candidates
            )
        )

        # -------------------------------------------------
        # Video segments
        # -------------------------------------------------

        video_segments = (
            self.group_video_segments(
                video_results
            )
        )

        return {

            "query_type":
                "image",

            "query_image":
                str(query_image),

            "image_candidates":
                image_candidates,

            "video_candidates":
                video_candidates,

            "image_results":
                image_results,

            "video_results":
                video_results,

            "video_segments":
                video_segments
        }


# =========================================================
# Main
# =========================================================

def main():

    searcher = (
        ImageQuerySemanticSearch()
    )

    query_image = (
        PROJECT_ROOT
        / "evidence"
        / "images"
        / "human1.jpg"
    )

    result = searcher.search(
        query_image
    )

    # =====================================================
    # Image Results
    # =====================================================

    print()
    print("=" * 70)
    print(
        "IMAGE QUERY → IMAGE RESULTS"
    )
    print("=" * 70)

    for rank, item in enumerate(
        result["image_results"],
        start=1
    ):

        print(
            f"[Rank {rank}] "
            f"{item['filename']}"
        )

        print(
            f"  FAISS     : "
            f"{item['faiss_score']:.4f}"
        )

        print(
            f"  Reranker  : "
            f"{item['rerank_score']:.4f}"
        )

    # =====================================================
    # Video Results
    # =====================================================

    print()
    print("=" * 70)
    print(
        "IMAGE QUERY → VIDEO RESULTS"
    )
    print("=" * 70)

    for rank, item in enumerate(
        result["video_results"],
        start=1
    ):

        print(
            f"[Rank {rank}] "
            f"{item['video']} "
            f"@ "
            f"{float(item['time']):.3f}s"
        )

        print(
            f"  Frame     : "
            f"{item['frame']}"
        )

        print(
            f"  FAISS     : "
            f"{item['faiss_score']:.4f}"
        )

        print(
            f"  Reranker  : "
            f"{item['rerank_score']:.4f}"
        )

    # =====================================================
    # Video Segments
    # =====================================================

    print()
    print("=" * 70)
    print(
        "IMAGE QUERY → VIDEO SEGMENTS"
    )
    print("=" * 70)

    for segment in result[
        "video_segments"
    ]:

        print(
            f"[Segment "
            f"{segment['segment_rank']}] "
            f"{segment['video']}"
        )

        print(
            f"  Time      : "
            f"{segment['start_time']:.3f}"
            f" ~ "
            f"{segment['end_time']:.3f}"
        )

        print(
            f"  Best      : "
            f"{segment['best_score']:.4f}"
        )

        print(
            f"  Average   : "
            f"{segment['average_score']:.4f}"
        )

        print(
            f"  Frames    : "
            f"{len(segment['frames'])}"
        )


if __name__ == "__main__":
    main()