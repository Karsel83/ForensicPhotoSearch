import os
import cv2

from ultralytics import YOLO

from reid_model import PersonReID
from similarity import cosine_similarity


class ImagePersonSearch:

    def __init__(
        self,
        model_path="yolo11n.pt",
        reid=None
    ):

        print("[*] Initializing Image Person Search")

        self.model = YOLO(model_path)

        if reid is None:
            self.reid = PersonReID()
        else:
            self.reid = reid

        print("[*] Image Person Search ready")

    # --------------------------------------------------
    # Person Crop
    # --------------------------------------------------

    def crop_person(
        self,
        image,
        bbox
    ):

        h, w = image.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))

        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[
            y1:y2,
            x1:x2
        ]

    # --------------------------------------------------
    # Detect persons
    # --------------------------------------------------

    def detect_persons(
        self,
        image
    ):

        results = self.model(
            image,
            verbose=False
        )

        persons = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls = int(
                    box.cls[0].item()
                )

                # COCO person class = 0
                if cls != 0:
                    continue

                confidence = float(
                    box.conf[0].item()
                )

                xyxy = box.xyxy[
                    0
                ].cpu().numpy()

                bbox = [
                    int(xyxy[0]),
                    int(xyxy[1]),
                    int(xyxy[2]),
                    int(xyxy[3])
                ]

                persons.append({
                    "bbox": bbox,
                    "confidence": confidence
                })

        return persons

    # --------------------------------------------------
    # Search one image
    # --------------------------------------------------

    def search_image(
        self,
        query_feature,
        image_path,
        evidence_dir
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:
            return []

        persons = self.detect_persons(
            image
        )

        results = []

        image_name = os.path.basename(
            image_path
        )

        image_stem = os.path.splitext(
            image_name
        )[0]

        for person_index, person in enumerate(
            persons
        ):

            bbox = person["bbox"]
            confidence = person["confidence"]
            crop = self.crop_person(
                image,
                bbox
            )

            if crop is None:
                continue

            # ------------------------------------------
            # Evidence directory
            # ------------------------------------------

            image_evidence_dir = os.path.join(
                evidence_dir,
                image_stem
            )

            os.makedirs(
                image_evidence_dir,
                exist_ok=True
            )

            crop_path = os.path.join(
                image_evidence_dir,
                f"person_{person_index}.jpg"
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            # ------------------------------------------
            # Re-ID
            # ------------------------------------------

            person_feature = self.reid.extract(
                crop_path
            )

            score = cosine_similarity(
                query_feature,
                person_feature
            )

            results.append({
                "source_type": "image",

                "image": image_name,

                "source": image_path,

                "filename": image_name,

                "person_index": person_index,

                "similarity": round(
                    float(score),
                    4
                ),

                "bbox": bbox,
                
                "confidence": confidence,

                "detection_confidence": confidence,

                "evidence_path": crop_path,

                "crop": crop_path
            })

        return results

    # --------------------------------------------------
    # Search image directory
    # --------------------------------------------------

    def search_directory(
        self,
        query_feature,
        image_dir,
        evidence_dir
    ):

        if not os.path.exists(
            image_dir
        ):
            print(
                f"[!] Image directory not found: "
                f"{image_dir}"
            )

            return []

        files = sorted(
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )
        )

        print(
            f"[*] Image files: {len(files)}"
        )

        results = []

        for index, filename in enumerate(
            files,
            start=1
        ):

            image_path = os.path.join(
                image_dir,
                filename
            )

            print(
                f"[{index:02d}/{len(files):02d}] "
                f"{filename}"
            )

            image_results = self.search_image(
                query_feature,
                image_path,
                evidence_dir
            )

            results.extend(
                image_results
            )

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return results