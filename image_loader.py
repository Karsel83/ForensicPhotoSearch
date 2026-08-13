from pathlib import Path
from PIL import Image
import hashlib


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def scan_images(folder):
    folder = Path(folder)

    results = []

    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(file_path) as img:

                width, height = img.size
                image_format = img.format

                image_info = {
                    "path": str(file_path),
                    "filename": file_path.name,
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "sha256": calculate_sha256(file_path)
                }

                results.append(image_info)

        except Exception as e:

            print(f"[ERROR] {file_path}")
            print(f"        {e}")

    return results