import os
import cv2
import sys

from .tracker import PersonTracker
from .evidence_manager import EvidenceManager


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from reid_model import PersonReID
from similarity import cosine_similarity


class VideoPersonSearch:

    def __init__(self):

        print(
            "[*] Initializing Video Person Search"
        )

        # 절대 경로 사용
        self.tracker = PersonTracker(
            model_path=os.path.join(
                PROJECT_ROOT,
                "yolo11n.pt"
            )
        )

        self.reid = PersonReID()

        self.evidence = EvidenceManager(
            evidence_root=os.path.join(
                PROJECT_ROOT,
                "video",
                "evidence"
            )
        )

        print(
            "[*] Video Person Search ready"
        )

    # ========================================================
    # Person Crop
    # ========================================================

    def crop_person(
        self,
        frame,
        bbox
    ):

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))

        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[
            y1:y2,
            x1:x2
        ]

    # ========================================================
    # Video Search
    #
    # query_feature:
    #   이미 계산된 Query OSNet embedding
    # ========================================================

    def search(
        self,
        query_feature,
        video_path,
        threshold=0.80,
        high_score_threshold=0.75,
        min_high_score_count=5,
        reid_interval=10
    ):

        tracks = {}

        print()
        print(
            f"[*] Video: {video_path}"
        )

        print(
            "[*] Starting video analysis..."
        )

        # ====================================================
        # Tracking
        # ====================================================

        for item in self.tracker.process(
            video_path
        ):

            frame_index = item["frame"]
            track_id = item["track_id"]
            bbox = item["bbox"]
            frame = item["frame_image"]
            fps = item["fps"]

            # ------------------------------------------------
            # 새로운 Track
            # ------------------------------------------------

            if track_id not in tracks:

                tracks[track_id] = {

                    "track_id":
                        track_id,

                    "first_frame":
                        frame_index,

                    "last_frame":
                        frame_index,

                    "scores":
                        [],

                    "best_score":
                        0.0,

                    "best_frame":
                        -1,

                    "best_bbox":
                        None,

                    "evidence_path":
                        None,

                    "high_score_count":
                        0
                }

            track = tracks[track_id]

            track["last_frame"] = frame_index

            # ------------------------------------------------
            # Re-ID 실행 간격
            # ------------------------------------------------

            if (
                frame_index % reid_interval
                != 0
            ):
                continue

            # ------------------------------------------------
            # Person Crop
            # ------------------------------------------------

            crop = self.crop_person(
                frame,
                bbox
            )

            if crop is None:
                continue

            # ------------------------------------------------
            # Evidence 저장
            # ------------------------------------------------

            track_dir = (
                self.evidence
                .create_track_directory(
                    track_id
                )
            )

            crop_path = os.path.join(
                track_dir,
                f"frame_{frame_index}.jpg"
            )

            success = cv2.imwrite(
                crop_path,
                crop
            )

            if not success:

                print(
                    f"[WARNING] "
                    f"Evidence 저장 실패: "
                    f"{crop_path}"
                )

                continue

            # ------------------------------------------------
            # Re-ID
            # ------------------------------------------------

            person_feature = (
                self.reid.extract(
                    crop_path
                )
            )

            # ------------------------------------------------
            # Similarity
            # ------------------------------------------------

            score = cosine_similarity(
                query_feature,
                person_feature
            )

            track["scores"].append(
                {
                    "frame": frame_index,
                    "score": float(score)
                }
            )

            # ------------------------------------------------
            # High Score Count
            # ------------------------------------------------

            if score >= high_score_threshold:

                track[
                    "high_score_count"
                ] += 1

            # ------------------------------------------------
            # Best Match
            # ------------------------------------------------

            if score > track["best_score"]:

                track["best_score"] = float(
                    score
                )

                track["best_frame"] = (
                    frame_index
                )

                track["best_bbox"] = list(
                    bbox
                )

                track["evidence_path"] = (
                    crop_path
                )

            print(
                f"[Frame {frame_index:05d}] "
                f"Track={track_id:<3} "
                f"Similarity={score:.4f}"
            )

        # ====================================================
        # Track 결과 정리
        # ====================================================

        results = []

        for track_id, track in tracks.items():

            scores = track["scores"]

            if not scores:
                continue

            # scores는
            # {"frame": ..., "score": ...}
            # 형태이므로 score 값만 추출
            score_values = [
                item["score"]
                for item in scores
            ]

            average_score = (
                sum(score_values)
                /
                len(score_values)
            )

            # --------------------------------------------
            # Track-level match 판정
            # --------------------------------------------

            match = (
                track["best_score"] >= threshold
                and
                track["high_score_count"]
                >= min_high_score_count
            )

            # --------------------------------------------
            # 결과 생성
            # --------------------------------------------

            result = {

                "track_id":
                    track_id,

                "match":
                    match,

                "best_score":
                    round(
                        track["best_score"],
                        4
                    ),

                "average_score":
                    round(
                        average_score,
                        4
                    ),

                "high_score_count":
                    track[
                        "high_score_count"
                    ],

                "sample_count":
                    len(scores),

                "first_frame":
                    track[
                        "first_frame"
                    ],

                "last_frame":
                    track[
                        "last_frame"
                    ],

                "best_frame":
                    track[
                        "best_frame"
                    ],

                "best_time":
                    round(
                        track[
                            "best_frame"
                        ] / fps,
                        3
                    ),

                "best_bbox":
                    track.get(
                        "best_bbox"
                    ),

                "evidence_path":
                    track.get(
                        "evidence_path"
                    ),

                "first_time":
                    round(
                        track[
                            "first_frame"
                        ] / fps,
                        3
                    ),

                "last_time":
                    round(
                        track[
                            "last_frame"
                        ] / fps,
                        3
                    ),

                "duration":
                    round(
                        (
                            track[
                                "last_frame"
                            ]
                            -
                            track[
                                "first_frame"
                            ]
                        ) / fps,
                        3
                    )
            }

            results.append(
                result
            )

        # ====================================================
        # Ranking
        # ====================================================

        results.sort(
            key=lambda x:
                x["best_score"],
            reverse=True
        )

        return results