from pathlib import Path
import json
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

PROJECT_ROOT = Path(__file__).resolve().parent

QWEN_ROOT = (
    PROJECT_ROOT
    / "Qwen3-VL-Embedding"
)

if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

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

IMAGE_INDEX = (
    PROJECT_ROOT
    / "data"
    / "semantic"
    / "image.index"
)

IMAGE_METADATA = (
    PROJECT_ROOT
    / "data"
    / "semantic"
    / "metadata.json"
)

VIDEO_INDEX = (
    PROJECT_ROOT
    / "data"
    / "semantic_video"
    / "video.index"
)

VIDEO_METADATA = (
    PROJECT_ROOT
    / "data"
    / "semantic_video"
    / "metadata.json"
)


# =========================================================
# Settings
# =========================================================

SAMPLE_INTERVAL_SECONDS = 1.0

IMAGE_CANDIDATE_K = 20
VIDEO_CANDIDATE_K = 20

FINAL_IMAGE_K = 10
FINAL_VIDEO_K = 10


# =========================================================
# Helpers
# =========================================================

def normalize(
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


# =========================================================
# Video Query Search
# =========================================================

class VideoQuerySemanticSearch:

    def __init__(
        self
    ):

        # -------------------------------------------------
        # Embedding
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

        self.image_index = faiss.read_index(
            str(IMAGE_INDEX)
        )

        self.image_metadata = load_json(
            IMAGE_METADATA
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

        self.video_index = faiss.read_index(
            str(VIDEO_INDEX)
        )

        self.video_metadata = load_json(
            VIDEO_METADATA
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
            "[+] Video query search ready"
        )

    # =====================================================
    # Sample Query Video
    # =====================================================

    def sample_video(
        self,
        video_path
    ):

        video_path = Path(
            video_path
        )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open video:\n"
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

        samples = []

        current_time = 0.0

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

            samples.append({
                "time":
                    round(
                        current_time,
                        3
                    ),

                "frame":
                    max(
                        int(
                            capture.get(
                                cv2.CAP_PROP_POS_FRAMES
                            )
                        ) - 1,
                        0
                    ),

                "image":
                    image
            })

            current_time += (
                SAMPLE_INTERVAL_SECONDS
            )

        capture.release()

        return samples

    # =====================================================
    # Query Video Embedding
    # =====================================================

    def encode_video(
        self,
        video_path
    ):

        samples = self.sample_video(
            video_path
        )

        if not samples:

            raise RuntimeError(
                "No frames were sampled "
                "from query video."
            )

        print(
            f"[*] Query frames: "
            f"{len(samples)}"
        )

        embeddings = []

        for index, sample in enumerate(
            samples,
            start=1
        ):

            print(
                f"[Query frame "
                f"{index:03d}/"
                f"{len(samples):03d}] "
                f"{sample['time']:.3f}s"
            )

            inputs = [{
                "image":
                    sample["image"],

                "instruction":
                    (
                        "Represent this image "
                        "for cross-modal retrieval."
                    )
            }]

            output = (
                self.embedding_model.process(
                    inputs
                )
            )

            if torch.is_tensor(
                output
            ):

                vector = (
                    output[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            else:

                vector = np.asarray(
                    output[0],
                    dtype=np.float32
                )

            embeddings.append(
                normalize(
                    vector
                )
            )

        # -------------------------------------------------
        # Mean pooling
        # -------------------------------------------------

        video_vector = np.mean(
            np.asarray(
                embeddings,
                dtype=np.float32
            ),
            axis=0
        )

        return normalize(
            video_vector
        )

    # =====================================================
    # Search Image
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
    # Search Video
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
    # Rerank Image
    # =====================================================

    def rerank_images(
        self,
        query_video_path,
        candidates
    ):

        # -------------------------------------------------
        # For first implementation:
        # use the strongest sampled query frame
        # as reranker representative.
        # -------------------------------------------------

        query_samples = self.sample_video(
            query_video_path
        )

        if not query_samples:
            return []

        query_frame = query_samples[0]["image"]

        temp_dir = (
            PROJECT_ROOT
            / "data"
            / "semantic_video"
            / "video_query_frames"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        query_frame_path = (
            temp_dir
            / "query_video_frame.jpg"
        )

        query_frame.save(
            query_frame_path
        )

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

                score = (
                    self.reranker
                    .score_image_to_image(
                        query_frame_path,
                        candidate["source"]
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

    def extract_frame(
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
                f"{frame_number}"
            )

        if not cv2.imwrite(
            str(output_path),
            frame
        ):
            raise RuntimeError(
                f"Failed to save:\n"
                f"{output_path}"
            )

        return output_path

    # =====================================================
    # Rerank Video
    # =====================================================

    def rerank_videos(
        self,
        query_video_path,
        candidates
    ):

        # -------------------------------------------------
        # Use a representative query frame initially.
        # Later this can be upgraded to multi-frame
        # query-to-candidate scoring.
        # -------------------------------------------------

        query_samples = self.sample_video(
            query_video_path
        )

        if not query_samples:
            return []

        query_frame_path = (
            PROJECT_ROOT
            / "data"
            / "semantic_video"
            / "video_query_frames"
            / "query_video_frame.jpg"
        )

        query_frame_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        query_samples[0]["image"].save(
            query_frame_path
        )

        results = []

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                f"[Video rerank "
                f"{index:02d}/"
                f"{len(candidates):02d}] "
                f"{candidate['video']} "
                f"frame={candidate['frame']}"
            )

            try:

                frame_path = (
                    PROJECT_ROOT
                    / "data"
                    / "semantic_video"
                    / "video_query_frames"
                    / (
                        f"{Path(candidate['source']).stem}"
                        f"_frame_{candidate['frame']}.jpg"
                    )
                )

                self.extract_frame(
                    candidate["source"],
                    candidate["frame"],
                    frame_path
                )

                score = (
                    self.reranker
                    .score_image_to_video(
                        query_frame_path,
                        frame_path
                    )
                )

                item = dict(
                    candidate
                )

                item["rerank_score"] = float(
                    score
                )

                item["rerank_frame"] = (
                    str(frame_path)
                )

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

    def group_segments(
        self,
        results,
        gap_seconds=1.0
    ):

        grouped = {}

        for result in results:

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

                t = float(
                    frame["time"]
                )

                score = float(
                    frame["rerank_score"]
                )

                if current is None:

                    current = {
                        "video":
                            video_name,

                        "source":
                            frame["source"],

                        "start_time":
                            t,

                        "end_time":
                            t,

                        "best_time":
                            t,

                        "best_score":
                            score,

                        "frames":
                            [frame]
                    }

                    continue

                gap = (
                    t
                    -
                    current["end_time"]
                )

                if gap <= gap_seconds:

                    current["end_time"] = t

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

                        current["best_time"] = t

                else:

                    current["duration"] = round(
                        current["end_time"]
                        -
                        current["start_time"],
                        3
                    )

                    current["average_score"] = (
                        np.mean([
                            item[
                                "rerank_score"
                            ]
                            for item in
                            current["frames"]
                        ])
                    )

                    segments.append(
                        current
                    )

                    current = {
                        "video":
                            video_name,

                        "source":
                            frame["source"],

                        "start_time":
                            t,

                        "end_time":
                            t,

                        "best_time":
                            t,

                        "best_score":
                            score,

                        "frames":
                            [frame]
                    }

            if current is not None:

                current["duration"] = round(
                    current["end_time"]
                    -
                    current["start_time"],
                    3
                )

                current["average_score"] = (
                    np.mean([
                        item["rerank_score"]
                        for item in
                        current["frames"]
                    ])
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

            segment["segment_rank"] = rank

        return segments

    # =====================================================
    # Full Search
    # =====================================================

    def search(
        self,
        query_video
    ):

        query_video = Path(
            query_video
        )

        if not query_video.exists():
            raise FileNotFoundError(
                f"Query video not found:\n"
                f"{query_video}"
            )

        print()
        print("=" * 70)
        print(
            "VIDEO → IMAGE + VIDEO SEARCH"
        )
        print("=" * 70)

        print(
            f"[*] Query video: "
            f"{query_video}"
        )

        # -------------------------------------------------
        # Stage 0: Query Video Embedding
        # -------------------------------------------------

        print()
        print(
            "[*] Building query video embedding..."
        )

        query_vector = self.encode_video(
            query_video
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
            "[*] Stage 2: Video → Image"
        )

        image_results = (
            self.rerank_images(
                query_video,
                image_candidates
            )
        )

        print()
        print(
            "[*] Stage 2: Video → Video"
        )

        video_results = (
            self.rerank_videos(
                query_video,
                video_candidates
            )
        )

        # -------------------------------------------------
        # Segments
        # -------------------------------------------------

        video_segments = (
            self.group_segments(
                video_results
            )
        )

        return {

            "query_type":
                "video",

            "query_video":
                str(query_video),

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
        VideoQuerySemanticSearch()
    )

    query_video = (
        PROJECT_ROOT
        / "video"
        / "data"
        / "test.mp4"
    )

    result = searcher.search(
        query_video
    )

    # =====================================================
    # Image Results
    # =====================================================

    print()
    print("=" * 70)
    print(
        "VIDEO QUERY → IMAGE RESULTS"
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
        "VIDEO QUERY → VIDEO RESULTS"
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
        "VIDEO QUERY → VIDEO SEGMENTS"
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
            f"  Time    : "
            f"{segment['start_time']:.3f}"
            f" ~ "
            f"{segment['end_time']:.3f}"
        )

        print(
            f"  Best    : "
            f"{segment['best_score']:.4f}"
        )

        print(
            f"  Average : "
            f"{segment['average_score']:.4f}"
        )

        print(
            f"  Frames  : "
            f"{len(segment['frames'])}"
        )


if __name__ == "__main__":
    main()