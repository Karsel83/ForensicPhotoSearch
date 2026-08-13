from pathlib import Path
from PIL import Image


class PersonCropper:

    def __init__(self, output_dir="data/person_crops"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crop(self, image_path, persons):

        image = Image.open(image_path).convert("RGB")

        crops = []

        for index, person in enumerate(persons):

            x1, y1, x2, y2 = person["bbox"]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))

            filename = Path(image_path).stem

            crop_path = (
                self.output_dir /
                f"{filename}_person_{index}.jpg"
            )

            crop.save(crop_path)

            crops.append({
                "crop_path": str(crop_path),
                "confidence": person["confidence"],
                "bbox": person["bbox"]
            })

        return crops