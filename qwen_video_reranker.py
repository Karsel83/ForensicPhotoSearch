from pathlib import Path
import sys

import torch
from PIL import Image


# =========================================================
# Project Root
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

RERANKER_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-vl"
    / "Qwen3-VL-Reranker-2B"
)

SCRIPTS_DIR = (
    RERANKER_PATH
    / "scripts"
)


# =========================================================
# Official Qwen Reranker wrapper
# =========================================================

if not SCRIPTS_DIR.exists():

    raise FileNotFoundError(
        "Qwen reranker scripts directory not found:\n"
        f"{SCRIPTS_DIR}\n\n"
        "Download the official scripts directory first."
    )


if str(SCRIPTS_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SCRIPTS_DIR)
    )


from qwen3_vl_reranker import (
    Qwen3VLReranker
)


# =========================================================
# Wrapper
# =========================================================

class QwenVideoReranker:

    def __init__(
        self,
        model_path=RERANKER_PATH
    ):

        print(
            "[*] Loading Qwen3-VL "
            "Reranker-2B..."
        )

        self.model = Qwen3VLReranker(
            model_name_or_path=
                str(model_path)
        )

        print(
            "[+] Qwen3-VL "
            "Reranker loaded"
        )


    # =====================================================
    # Text → Image reranking
    # =====================================================

    def score_image(
        self,
        query,
        image
    ):

        if isinstance(
            image,
            (str, Path)
        ):

            image = str(
                image
            )

        inputs = {

            "instruction":
                (
                    "Given a search query, "
                    "retrieve relevant images "
                    "that answer the query."
                ),

            "query": {
                "text":
                    query
            },

            "documents": [
                {
                    "image":
                        image
                }
            ]
        }

        scores = self.model.process(
            inputs
        )

        if not scores:

            raise RuntimeError(
                "Qwen reranker returned no score."
            )

        score = scores[0]

        if torch.is_tensor(
            score
        ):

            return float(
                score
                .detach()
                .float()
                .cpu()
                .item()
            )

        return float(
            score
        )


    # =====================================================
    # Text → Video reranking
    # =====================================================

    def score_video(
        self,
        query,
        video
    ):

        if isinstance(
            video,
            (str, Path)
        ):

            video = str(
                video
            )

        inputs = {

            "instruction":
                (
                    "Given a search query, "
                    "retrieve relevant video content "
                    "that answers the query."
                ),

            "query": {
                "text":
                    query
            },

            "documents": [
                {
                    "video":
                        video
                }
            ]
        }

        scores = self.model.process(
            inputs
        )

        if not scores:

            raise RuntimeError(
                "Qwen reranker returned no score."
            )

        score = scores[0]

        if torch.is_tensor(
            score
        ):

            return float(
                score
                .detach()
                .float()
                .cpu()
                .item()
            )

        return float(
            score
        )
        # =====================================================
    # Image → Image reranking
    # =====================================================

    def score_image_to_image(
        self,
        query_image,
        candidate_image
    ):

        if isinstance(
            query_image,
            (str, Path)
        ):

            query_image = str(
                query_image
            )

        if isinstance(
            candidate_image,
            (str, Path)
        ):

            candidate_image = str(
                candidate_image
            )

        inputs = {

            "instruction":
                (
                    "Given a query image, "
                    "retrieve relevant images."
                ),

            "query": {
                "image":
                    query_image
            },

            "documents": [
                {
                    "image":
                        candidate_image
                }
            ]
        }

        scores = self.model.process(
            inputs
        )

        if not scores:

            raise RuntimeError(
                "Qwen reranker returned no score."
            )

        score = scores[0]

        if torch.is_tensor(
            score
        ):

            return float(
                score
                .detach()
                .float()
                .cpu()
                .item()
            )

        return float(score)


    # =====================================================
    # Image → Video Frame reranking
    # =====================================================

    def score_image_to_video(
        self,
        query_image,
        candidate_frame
    ):

        if isinstance(
            query_image,
            (str, Path)
        ):

            query_image = str(
                query_image
            )

        if isinstance(
            candidate_frame,
            (str, Path)
        ):

            candidate_frame = str(
                candidate_frame
            )

        inputs = {

            "instruction":
                (
                    "Given a query image, "
                    "retrieve relevant video frames."
                ),

            "query": {
                "image":
                    query_image
            },

            "documents": [
                {
                    "image":
                        candidate_frame
                }
            ]
        }

        scores = self.model.process(
            inputs
        )

        if not scores:

            raise RuntimeError(
                "Qwen reranker returned no score."
            )

        score = scores[0]

        if torch.is_tensor(
            score
        ):

            return float(
                score
                .detach()
                .float()
                .cpu()
                .item()
            )

        return float(score)