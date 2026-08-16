import os
import shutil
from pathlib import Path

from image_loader import scan_images
from image_database import save_database
from person_detector import PersonDetector
from person_cropper import PersonCropper


IMAGE_FOLDER = "evidence/images"
DATABASE_FILE = "data/images.json"


def main():

    print("=" * 60)
    print("Forensic Photo Search")
    print("=" * 60)

    image_root = Path(IMAGE_FOLDER).resolve()

    # Evidence crops are stored below evidence/images/<image-name>/.
    # Only top-level files are source evidence images.
    images = [
        image for image in scan_images(IMAGE_FOLDER)
        if Path(image["path"]).resolve().parent == image_root
    ]

    print(f"[*] 발견한 이미지: {len(images)}개")

    detector = PersonDetector()
    cropper = PersonCropper()

    print()
    print("[*] 사람 검출 및 Crop 시작...")

    for image in images:

        persons = detector.detect(
            image["path"]
        )

        image["person_count"] = len(persons)
        image["persons"] = persons

        crops = cropper.crop(
            image["path"],
            persons
        )

        evidence_dir = Path(IMAGE_FOLDER) / Path(image["path"]).stem

        for person_index, crop in enumerate(crops):
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence_path = evidence_dir / f"person_{person_index}.jpg"
            shutil.copy2(crop["crop_path"], evidence_path)

            crop["person_index"] = person_index
            crop["evidence_path"] = str(evidence_path)

        image["person_crops"] = crops

        print(
            f"{image['filename']:30}"
            f"person={len(persons)} "
            f"crop={len(crops)}"
        )

    save_database(
        images,
        DATABASE_FILE
    )

    print()
    print("[*] 분석 결과 저장 완료")


if __name__ == "__main__":
    main()
