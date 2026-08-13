import os
import cv2
import sys

from tracker import PersonTracker
from evidence_manager import EvidenceManager


# -----------------------------------------
# ForensicPhotoSearch 상위 폴더를 import 경로에 추가
# -----------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from reid_model import PersonReID
from similarity import cosine_similarity


class VideoPersonSearch:

    def __init__(self):

        print(
            "[*] Initializing Video Person Search"
        )

        self.tracker = PersonTracker(
            model_path="../yolo11n.pt"
        )

        self.reid = PersonReID()

        self.evidence = EvidenceManager(
            evidence_root="evidence"
        )

        print(
            "[*] Video Person Search ready"
        )

    # -----------------------------------------
    # Person Crop
    # -----------------------------------------

    def crop_person(
        self,
        frame,
        bbox
    ):

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            w,
            x2
        )

        y2 = min(
            h,
            y2
        )

        if x2 <= x1 or y2 <= y1:

            return None

        crop = frame[
            y1:y2,
            x1:x2
        ]

        return crop

    # -----------------------------------------
    # Query Image → OSNet Feature
    # -----------------------------------------

    def extract_query_feature(
        self,
        query_path
    ):

        print(
            f"[*] Query image: {query_path}"
        )

        feature = self.reid.extract(
            query_path
        )

        print(
            f"[*] Query feature shape: "
            f"{feature.shape}"
        )

        return feature

    # -----------------------------------------
    # Video Search
    # -----------------------------------------

    def search(
        self,
        query_path,
        video_path,
        evidence_dir,
        threshold=0.80,
        high_score_threshold=0.75,
        min_high_score_count=5,
        reid_interval=10
    ):

        os.makedirs(
            evidence_dir,
            exist_ok=True
        )

        # =====================================
        # 1. Query Feature
        # =====================================

        query_feature = (
            self.extract_query_feature(
                query_path
            )
        )

        # =====================================
        # 2. Track Information
        # =====================================

        tracks = {}

        print()

        print(
            "[*] Starting video analysis..."
        )

        # =====================================
        # 3. Video Tracking
        # =====================================

        for item in self.tracker.process(
            video_path
        ):

            frame_index = item["frame"]

            track_id = item["track_id"]

            bbox = item["bbox"]

            frame = item["frame_image"]

            fps = item["fps"]

            # ---------------------------------
            # 새로운 Track
            # ---------------------------------

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

            # 마지막 등장 프레임 갱신

            track["last_frame"] = (
                frame_index
            )

            # ---------------------------------
            # Re-ID 실행 간격
            # ---------------------------------

            if (
                frame_index
                % reid_interval
                != 0
            ):

                continue

            # =================================
            # 4. Person Crop
            # =================================

            crop = self.crop_person(
                frame,
                bbox
            )

            if crop is None:

                continue

            # =================================
            # 5. Track Evidence Crop 저장
            # =================================

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

            # =================================
            # 6. OSNet Feature
            # =================================

            person_feature = (
                self.reid.extract(
                    crop_path
                )
            )

            # =================================
            # 7. Cosine Similarity
            # =================================

            score = cosine_similarity(
                query_feature,
                person_feature
            )

            track["scores"].append(
                score
            )

            # =================================
            # 8. High Score Count
            # =================================

            if (
                score
                >= high_score_threshold
            ):

                track[
                    "high_score_count"
                ] += 1

            # =================================
            # 9. Best Match
            # =================================

            if (
                score
                > track["best_score"]
            ):

                track["best_score"] = (
                    score
                )

                track["best_frame"] = (
                    frame_index
                )

                track["best_bbox"] = (
                    list(bbox)
                )

                track["evidence_path"] = (
                    crop_path
                )

            # ---------------------------------
            # 현재 결과 출력
            # ---------------------------------

            print(
                f"[Frame {frame_index:05d}] "
                f"Track={track_id:<3} "
                f"Similarity={score:.4f}"
            )

        # =====================================
        # 10. 결과 정리
        # =====================================

        results = []

        for (
            track_id,
            track
        ) in tracks.items():

            scores = track["scores"]

            if not scores:

                continue

            # ---------------------------------
            # 평균 Similarity
            # ---------------------------------

            average_score = (
                sum(scores)
                /
                len(scores)
            )

            # ---------------------------------
            # Match 판정
            # ---------------------------------

            match = (

                track["best_score"]
                >= threshold

                and

                track["high_score_count"]
                >= min_high_score_count
            )

            # =================================
            # Evidence 생성
            # =================================

            evidence_paths = (
                self.evidence
                .save_context_evidence(
                    track_id=track_id,
                    best_frame=(
                        track["best_frame"]
                    )
                )
            )

            # =================================
            # 시간 계산
            # =================================

            first_time = round(
                track["first_frame"]
                / fps,
                3
            )

            last_time = round(
                track["last_frame"]
                / fps,
                3
            )

            best_time = round(
                track["best_frame"]
                / fps,
                3
            )

            # =================================
            # Metadata
            # =================================

            metadata = {

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
                    best_time,

                "first_time":
                    first_time,

                "last_time":
                    last_time,

                "best_bbox":
                    track[
                        "best_bbox"
                    ],

                "evidence":
                    evidence_paths
            }

            metadata_path = (
                self.evidence
                .save_metadata(
                    track_id,
                    metadata
                )
            )

            # =================================
            # 결과
            # =================================

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

                "best_bbox":
                    track[
                        "best_bbox"
                    ],

                "evidence_path":
                    track[
                        "evidence_path"
                    ],

                "evidence":
                    evidence_paths,

                "metadata_path":
                    metadata_path,

                "first_time":
                    first_time,

                "last_time":
                    last_time,

                "best_time":
                    best_time
            }

            results.append(
                result
            )

        # =====================================
        # 11. Ranking
        # =====================================

        results.sort(
            key=lambda x:
                x["best_score"],
            reverse=True
        )

        return results