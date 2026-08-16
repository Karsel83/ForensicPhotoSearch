import os
import cv2

from ultralytics import YOLO

from reid_model import PersonReID
from similarity import cosine_similarity
from image_database import ImageEmbeddingIndex
from two_stage_faiss import TwoStageImageSearch


class ImagePersonSearch:

    def __init__(
        self,
        model_path="yolo11n.pt",
        reid=None,
        embeddings_path="data/embeddings/embeddings.npy",
        metadata_path="data/embeddings/metadata.json",
        faiss_index_path="data/faiss/image.index",
        candidate_k=1000,
        top_k=50
    ):

        print("[*] Initializing Image Person Search")

        self.model_path = model_path
        self.model = None
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.faiss_index_path = faiss_index_path
        self.candidate_k = candidate_k
        self.top_k = top_k

        # The unified search supplies the already-loaded query Re-ID model.
        # Keep standalone/legacy use lazy so vector-only search never loads it.
        self.reid = reid

        print("[*] Image Person Search ready (vector index mode)")

    def _get_detector(self):
        """Kept for the legacy single-image helper; directory search uses no YOLO."""
        if self.model is None:
            self.model = YOLO(self.model_path)
        return self.model

    def _get_reid(self):
        if self.reid is None:
            self.reid = PersonReID()
        return self.reid

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

        results = self._get_detector()(
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

            person_feature = self._get_reid().extract(
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

        # FAISS retrieves candidates; exact cosine re-ranks only those candidates.
        # The compatible arguments remain because forensic_search.py already calls this method.
        searcher = TwoStageImageSearch(
            index_file=self.faiss_index_path,
            embeddings_file=self.embeddings_path,
            metadata_file=self.metadata_path
        )

        results, metrics = searcher.search(
            query_feature,
            candidate_k=self.candidate_k,
            top_k=self.top_k
        )

        print(
            f"[*] Two-stage image search: "
            f"candidates={metrics['candidate_k']}, "
            f"total={metrics['total_ms']:.4f}ms"
        )

        return results
