from ultralytics import YOLO


class PersonDetector:

    def __init__(self):
        print("[*] YOLO 모델 로딩 중...")

        self.model = YOLO("models/yolo11n.pt")

        print("[*] YOLO 모델 로딩 완료")

    def detect(self, image_path):

        results = self.model(
            image_path,
            verbose=False
        )

        persons = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # COCO class 0 = person
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                persons.append({
                    "confidence": confidence,
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                })

        return persons