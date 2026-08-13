def format_image_result(result):
    """
    ImagePersonSearch 결과를
    통합 결과 형식으로 변환
    """

    return {
        "type": "image",

        "score": round(
            float(result["similarity"]),
            4
        ),

        "source": result["image"],

        "person_index": result.get(
            "person_index"
        ),

        "track_id": None,

        "frame": None,

        "time": None,

        "start_time": None,

        "end_time": None,

        "bbox": result.get(
            "bbox"
        ),

        "detection_confidence": result.get(
            "detection_confidence"
        ),

        "evidence": result.get(
            "evidence_path"
        )
    }


def format_video_result(
    result,
    video_name
):
    """
    VideoPersonSearch 결과를
    통합 결과 형식으로 변환
    """

    return {
        "type": "video",

        "score": round(
            float(result["best_score"]),
            4
        ),

        "source": video_name,

        "person_index": None,

        "track_id": result.get(
            "track_id"
        ),

        "frame": result.get(
            "best_frame"
        ),

        "time": result.get(
            "best_time"
        ),

        "start_time": result.get(
            "first_time"
        ),

        "end_time": result.get(
            "last_time"
        ),

        "bbox": result.get(
            "best_bbox"
        ),

        "detection_confidence": None,

        "evidence": result.get(
            "evidence_path"
        )
    }


def merge_results(
    image_results,
    video_results,
    video_name
):
    """
    Image + Video 결과를 하나로 합치고
    similarity score 기준으로 정렬
    """

    unified = []

    # -------------------------
    # Image
    # -------------------------

    for result in image_results:

        unified.append(
            format_image_result(
                result
            )
        )

    # -------------------------
    # Video
    # -------------------------

    for result in video_results:

        unified.append(
            format_video_result(
                result,
                video_name
            )
        )

    # -------------------------
    # Score 정렬
    # -------------------------

    unified.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # -------------------------
    # Rank 추가
    # -------------------------

    for rank, result in enumerate(
        unified,
        start=1
    ):

        result["rank"] = rank

    return unified