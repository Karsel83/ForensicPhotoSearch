from pathlib import Path
import json
import os
import sys
import cv2
import numpy as np
import torch

# =========================================================
# Project
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

QWEN_ROOT = PROJECT_ROOT / "Qwen3-VL-Embedding"

if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_video_reranker import QwenVideoReranker


# =========================================================
# Paths
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-vl"
    / "Qwen3-VL-Embedding-2B"
)

VIDEO_INDEX_DIR = (
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

PROJECT_VIDEO_INDEX_FILE = (
    VIDEO_INDEX_DIR
    / "video.index"
)

FALLBACK_VIDEO_INDEX_FILE = (
    FAISS_ROOT
    / "semantic_video"
    / "video.index"
)

VIDEO_INDEX_FILE = (
    FALLBACK_VIDEO_INDEX_FILE
    if FALLBACK_VIDEO_INDEX_FILE.exists()
    else PROJECT_VIDEO_INDEX_FILE
)

VIDEO_METADATA_FILE = (
    VIDEO_INDEX_DIR
    / "metadata.json"
)


# =========================================================
# Settings
# =========================================================

CANDIDATE_K = 20
FINAL_TOP_K = 10

GROUP_GAP_SECONDS = 1.0
MIN_RERANK_SCORE = 0.0


# =========================================================
# Search
# =========================================================

class SemanticVideoSearch:

    def __init__(self):

        print("[*] Loading Qwen3-VL Embedding...")

        self.embedding_model = Qwen3VLEmbedder(
            model_name_or_path=str(MODEL_PATH)
        )

        print("[+] Embedding model loaded")

        print("[*] Loading Qwen3-VL Reranker...")

        self.reranker = QwenVideoReranker()

        print("[+] Reranker loaded")

        print("[*] Loading semantic video FAISS...")

        self.index = faiss_read_index(
            VIDEO_INDEX_FILE
        )

        with open(
            VIDEO_METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata length mismatch."
            )

        print(
            f"[+] Video index loaded: "
            f"{self.index.ntotal} vectors"
        )

    # =====================================================
    # Text Embedding
    # =====================================================

    def encode_text(self, text):

        inputs = [{
            "text": text,
            "instruction":
                "Retrieve video frames relevant to the user's query."
        }]

        embedding = self.embedding_model.process(
            inputs
        )

        if torch.is_tensor(embedding):

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

        vector = vector.astype(
            np.float32
        ).reshape(-1)

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector /= norm

        return vector

    # =====================================================
    # Stage 1: FAISS
    # =====================================================

    def retrieve_candidates(
        self,
        query
    ):

        query_vector = self.encode_text(
            query
        ).reshape(
            1,
            -1
        ).astype(
            np.float32
        )

        scores, ids = self.index.search(
            query_vector,
            min(
                CANDIDATE_K,
                self.index.ntotal
            )
        )

        candidates = []

        for rank, (score, vector_id) in enumerate(
            zip(
                scores[0],
                ids[0]
            ),
            start=1
        ):

            if vector_id < 0:
                continue

            item = dict(
                self.metadata[
                    int(vector_id)
                ]
            )

            item["faiss_rank"] = rank
            item["faiss_score"] = float(score)
            item["vector_id"] = int(vector_id)

            candidates.append(item)

        return candidates

    # =====================================================
    # Extract actual frame
    # =====================================================

    def extract_frame(
        self,
        video_path,
        frame_number
    ):

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            return None

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )

        success, frame = capture.read()

        capture.release()

        if not success:
            return None

        return frame

    # =====================================================
    # Stage 2: Reranker
    # =====================================================

    def rerank_candidates(
        self,
        query,
        candidates
    ):

        reranked = []

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                f"[Rerank {index:02d}/"
                f"{len(candidates):02d}] "
                f"{candidate['video']} "
                f"frame={candidate['frame']}"
            )

            source = Path(
                candidate["source"]
            )

            if not source.exists():
                source = (
                    PROJECT_ROOT
                    / "video"
                    / "data"
                    / Path(
                        candidate["video"]
                    ).name
                )

            frame_number = int(
                candidate["frame"]
            )

            if not source.exists():
                print(
                    f"[!] Video not found: {source}"
                )
                continue

            frame = self.extract_frame(
                source,
                frame_number
            )

            if frame is None:
                print(
                    "[!] Frame extraction failed."
                )
                continue

            # Save temporary frame.
            temp_dir = (
                FAISS_ROOT
                / "rerank_frames"
            )

            temp_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            temp_path = (
                temp_dir
                / (
                    f"{Path(source).stem}"
                    f"_frame_{frame_number}.jpg"
                )
            )

            cv2.imwrite(
                str(temp_path),
                frame
            )

            try:

                score = self.reranker.score_image(
                    query,
                    temp_path
                )

            except Exception as error:

                print(
                    f"[!] Reranker failed: "
                    f"{error}"
                )
                continue

            item = dict(candidate)

            item["rerank_score"] = round(
                float(score),
                6
            )

            reranked.append(item)

        reranked.sort(
            key=lambda x:
                x["rerank_score"],
            reverse=True
        )

        return reranked[:FINAL_TOP_K]

    # =====================================================
    # Segment grouping
    # =====================================================

    def group_segments(
        self,
        results
    ):

        grouped = {}

        for result in results:

            video = result.get(
                "video"
            )

            if video is None:
                continue

            grouped.setdefault(
                video,
                []
            ).append(
                result
            )

        segments = []

        for video, frames in grouped.items():

            frames.sort(
                key=lambda x:
                    float(
                        x["time"]
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
                        "video": video,
                        "source": frame["source"],
                        "start_time": t,
                        "end_time": t,
                        "best_time": t,
                        "best_frame": frame["frame"],
                        "best_score": score,
                        "average_score": score,
                        "frames": [frame]
                    }

                    continue

                gap = (
                    t
                    -
                    current["end_time"]
                )

                if gap <= GROUP_GAP_SECONDS:

                    current["end_time"] = t

                    current["frames"].append(
                        frame
                    )

                    if score > current["best_score"]:

                        current["best_score"] = score
                        current["best_time"] = t
                        current["best_frame"] = frame["frame"]

                    current["average_score"] = (
                        sum(
                            f["rerank_score"]
                            for f in current["frames"]
                        )
                        /
                        len(current["frames"])
                    )

                else:

                    current["duration"] = round(
                        current["end_time"]
                        - current["start_time"],
                        3
                    )

                    segments.append(
                        current
                    )

                    current = {
                        "video": video,
                        "source": frame["source"],
                        "start_time": t,
                        "end_time": t,
                        "best_time": t,
                        "best_frame": frame["frame"],
                        "best_score": score,
                        "average_score": score,
                        "frames": [frame]
                    }

            if current is not None:

                current["duration"] = round(
                    current["end_time"]
                    - current["start_time"],
                    3
                )

                segments.append(
                    current
                )

        segments.sort(
            key=lambda x:
                x["best_score"],
            reverse=True
        )

        for rank, segment in enumerate(
            segments,
            start=1
        ):
            segment["rank"] = rank

        return segments

    # =====================================================
    # Full Search
    # =====================================================

    def search(
        self,
        query
    ):

        print()
        print("=" * 60)
        print("TEXT → VIDEO 2-STAGE SEARCH")
        print("=" * 60)

        print(
            f"[*] Query: {query}"
        )

        # Stage 1
        candidates = self.retrieve_candidates(
            query
        )

        print(
            f"[*] Stage 1 candidates: "
            f"{len(candidates)}"
        )

        # Stage 2
        reranked = self.rerank_candidates(
            query,
            candidates
        )

        print(
            f"[*] Stage 2 results: "
            f"{len(reranked)}"
        )

        # Segment
        segments = self.group_segments(
            reranked
        )

        return {
            "query": query,
            "candidates": candidates,
            "reranked": reranked,
            "segments": segments
        }


# =========================================================
# FAISS helper
# =========================================================

def faiss_read_index(
    path
):

    import faiss

    return faiss.read_index(
        str(path)
    )


# =========================================================
# Main
# =========================================================

def main():

    searcher = SemanticVideoSearch()

    result = searcher.search(
        "빨간 패딩을 입은 남자"
    )

    print()
    print("=" * 60)
    print("FINAL RERANKED RESULTS")
    print("=" * 60)

    for item in result["reranked"]:

        print(
            f"[Rank {item.get('faiss_rank')}] "
            f"{item['video']} "
            f"{item['time']:.3f}s"
        )

        print(
            f"  FAISS     : "
            f"{item['faiss_score']:.4f}"
        )

        print(
            f"  Reranker  : "
            f"{item['rerank_score']:.4f}"
        )

    print()
    print("=" * 60)
    print("FINAL SEMANTIC SEGMENTS")
    print("=" * 60)

    for segment in result["segments"]:

        print(
            f"[Rank {segment['rank']}] "
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