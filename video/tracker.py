import cv2
from ultralytics import YOLO


class PersonTracker:

    def __init__(
        self,
        model_path="../yolo11n.pt"
    ):

        print(
            f"[*] Loading YOLO: {model_path}"
        )

        self.model = YOLO(
            model_path
        )

        print(
            "[*] YOLO loaded"
        )

    def process(
        self,
        video_path
    ):

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"영상을 열 수 없습니다: "
                f"{video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_index = 0

        print(
            f"[*] Video FPS: {fps}"
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.track(

                frame,

                persist=True,

                tracker="botsort.yaml",

                classes=[0],

                conf=0.4,

                verbose=False
            )

            result = results[0]

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                boxes = (
                    result.boxes
                    .xyxy
                    .cpu()
                    .numpy()
                )

                track_ids = (

                    result.boxes
                    .id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                for box, track_id in zip(
                    boxes,
                    track_ids
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    yield {

                        "frame":
                            frame_index,

                        "track_id":
                            int(track_id),

                        "bbox":
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            ),

                        "frame_image":
                            frame,

                        "fps":
                            fps
                    }

            frame_index += 1

        cap.release()

        print(
            "[*] Tracking finished"
        )