from typing import Iterable

def rank_score(
    rank: int,
    total: int
) -> float:

    if total <= 1:
        return 1.0

    rank = max(
        1,
        min(rank, total)
    )

    return 1.0 - (
        (rank - 1)
        / (total - 1)
    )


def add_rank_scores(
    results: list,
    score_key: str
) -> list:

    ordered = sorted(
        results,
        key=lambda item:
            float(
                item.get(
                    score_key,
                    0.0
                )
            ),
        reverse=True
    )

    total = len(ordered)

    for rank, item in enumerate(
        ordered,
        start=1
    ):

        item["source_rank"] = rank

        item["rank_score"] = round(
            rank_score(
                rank,
                total
            ),
            6
        )

    return ordered


def fuse_modalities(
    image_results: list,
    video_results: list,
    image_weight: float = 0.5,
    video_weight: float = 0.5
) -> list:

    if image_weight < 0:
        raise ValueError(
            "image_weight must be >= 0"
        )

    if video_weight < 0:
        raise ValueError(
            "video_weight must be >= 0"
        )

    weight_sum = (
        image_weight
        +
        video_weight
    )

    if weight_sum <= 0:
        raise ValueError(
            "At least one modality weight "
            "must be greater than 0."
        )

    image_weight /= weight_sum
    video_weight /= weight_sum

    # -----------------------------------------------------
    # IMAGE
    # Current semantic image search returns
    # "similarity"
    # -----------------------------------------------------

    image_results = add_rank_scores(
        image_results,
        "similarity"
    )

    # -----------------------------------------------------
    # VIDEO
    # Current semantic video search returns
    # "rerank_score"
    # -----------------------------------------------------

    video_results = add_rank_scores(
        video_results,
        "rerank_score"
    )

    unified = []

    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    for item in image_results:

        result = dict(
            item
        )

        result["result_type"] = "image"

        result["raw_score"] = float(
            item.get(
                "similarity",
                0.0
            )
        )

        result["fusion_score"] = round(
            image_weight
            *
            item["rank_score"],
            6
        )

        unified.append(
            result
        )

    # -----------------------------------------------------
    # Video
    # -----------------------------------------------------

    for item in video_results:

        result = dict(
            item
        )

        result["result_type"] = "video"

        result["raw_score"] = float(
            item.get(
                "rerank_score",
                0.0
            )
        )

        result["fusion_score"] = round(
            video_weight
            *
            item["rank_score"],
            6
        )

        unified.append(
            result
        )

    # -----------------------------------------------------
    # Unified ranking
    # -----------------------------------------------------

    unified.sort(
        key=lambda item:
            item["fusion_score"],
        reverse=True
    )

    for final_rank, item in enumerate(
        unified,
        start=1
    ):

        item["unified_rank"] = final_rank

    return unified