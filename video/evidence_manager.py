import os
import json
import cv2


class EvidenceManager:

    def __init__(self, evidence_root="evidence"):

        self.evidence_root = evidence_root

        os.makedirs(
            self.evidence_root,
            exist_ok=True
        )

    # -----------------------------------------
    # Track Evidence 폴더
    # -----------------------------------------

    def create_track_directory(self, track_id):

        track_dir = os.path.join(
            self.evidence_root,
            f"track_{track_id}"
        )

        os.makedirs(
            track_dir,
            exist_ok=True
        )

        return track_dir

    # -----------------------------------------
    # 이미지 저장
    # -----------------------------------------

    def save_frame(
        self,
        frame,
        track_id,
        filename
    ):

        track_dir = self.create_track_directory(
            track_id
        )

        path = os.path.join(
            track_dir,
            filename
        )

        success = cv2.imwrite(
            path,
            frame
        )

        if not success:

            raise RuntimeError(
                f"Evidence frame 저장 실패: {path}"
            )

        return path

    # -----------------------------------------
    # 이미 저장된 crop 중
    # best frame 주변 프레임 찾기
    # -----------------------------------------

    def find_context_frames(
        self,
        track_id,
        best_frame
    ):

        track_dir = self.create_track_directory(
            track_id
        )

        frame_files = []

        for filename in os.listdir(track_dir):

            if not filename.startswith("frame_"):
                continue

            if not filename.endswith(".jpg"):
                continue

            try:

                frame_number = int(
                    filename[
                        len("frame_"):-4
                    ]
                )

            except ValueError:

                continue

            frame_files.append(
                (
                    frame_number,
                    os.path.join(
                        track_dir,
                        filename
                    )
                )
            )

        frame_files.sort(
            key=lambda x: x[0]
        )

        before = None
        best = None
        after = None

        # -------------------------------------
        # Best frame
        # -------------------------------------

        for frame_number, path in frame_files:

            if frame_number == best_frame:

                best = path

                break

        # -------------------------------------
        # Before / After
        # -------------------------------------

        if best is not None:

            before_candidates = [
                (frame_number, path)
                for frame_number, path
                in frame_files
                if frame_number < best_frame
            ]

            after_candidates = [
                (frame_number, path)
                for frame_number, path
                in frame_files
                if frame_number > best_frame
            ]

            if before_candidates:

                before = before_candidates[-1][1]

            if after_candidates:

                after = after_candidates[0][1]

        return {
            "before": before,
            "best": best,
            "after": after
        }

    # -----------------------------------------
    # Best / Before / After를
    # 사람이 보기 좋은 이름으로 저장
    # -----------------------------------------

    def save_context_evidence(
        self,
        track_id,
        best_frame
    ):

        context = self.find_context_frames(
            track_id,
            best_frame
        )

        track_dir = self.create_track_directory(
            track_id
        )

        result = {}

        # -------------------------------------
        # Before
        # -------------------------------------

        if context["before"] is not None:

            before_image = cv2.imread(
                context["before"]
            )

            before_path = os.path.join(
                track_dir,
                "before.jpg"
            )

            cv2.imwrite(
                before_path,
                before_image
            )

            result["before_frame"] = (
                before_path
            )

        # -------------------------------------
        # Best
        # -------------------------------------

        if context["best"] is not None:

            best_image = cv2.imread(
                context["best"]
            )

            best_path = os.path.join(
                track_dir,
                "best_frame.jpg"
            )

            cv2.imwrite(
                best_path,
                best_image
            )

            result["best_frame"] = (
                best_path
            )

        # -------------------------------------
        # After
        # -------------------------------------

        if context["after"] is not None:

            after_image = cv2.imread(
                context["after"]
            )

            after_path = os.path.join(
                track_dir,
                "after.jpg"
            )

            cv2.imwrite(
                after_path,
                after_image
            )

            result["after_frame"] = (
                after_path
            )

        return result

    # -----------------------------------------
    # Metadata 저장
    # -----------------------------------------

    def save_metadata(
        self,
        track_id,
        metadata
    ):

        track_dir = self.create_track_directory(
            track_id
        )

        metadata_path = os.path.join(
            track_dir,
            "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False
            )

        return metadata_path