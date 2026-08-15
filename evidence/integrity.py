import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


class EvidenceIntegrity:

    def __init__(self, manifest_path="evidence/integrity/manifest.json"):
        self.manifest_path = Path(manifest_path)

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # MD5 / SHA-256
    # =========================================================

    def calculate_hashes(self, file_path):

        md5 = hashlib.md5()
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                md5.update(chunk)
                sha256.update(chunk)

        return {
            "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest()
        }

    # =========================================================
    # pHash
    # =========================================================

    def calculate_phash(self, file_path):

        image = cv2.imread(
            str(file_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"이미지를 읽을 수 없습니다: {file_path}"
            )

        resized = cv2.resize(
            image,
            (32, 32),
            interpolation=cv2.INTER_AREA
        )

        resized = np.float32(
            resized
        )

        dct = cv2.dct(
            resized
        )

        low_freq = dct[:8, :8]

        coefficients = low_freq.flatten()[1:]

        median = np.median(
            coefficients
        )

        bits = (
            coefficients > median
        )

        return "".join(
            "1" if bit else "0"
            for bit in bits
        )

    # =========================================================
    # pHash Hamming Distance
    # =========================================================

    @staticmethod
    def phash_distance(
        phash_a,
        phash_b
    ):

        if len(phash_a) != len(phash_b):
            raise ValueError(
                "pHash 길이가 다릅니다."
            )

        return sum(
            a != b
            for a, b in zip(
                phash_a,
                phash_b
            )
        )

    # =========================================================
    # Evidence Record
    # =========================================================

    def create_record(
        self,
        file_path,
        evidence_id,
        source_type="original"
    ):

        path = Path(
            file_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {path}"
            )

        hashes = self.calculate_hashes(
            path
        )

        phash = self.calculate_phash(
            path
        )

        return {

            "evidence_id":
                evidence_id,

            "file_path":
                str(
                    path.resolve()
                ),

            "filename":
                path.name,

            "source_type":
                source_type,

            "size":
                path.stat().st_size,

            "md5":
                hashes["md5"],

            "sha256":
                hashes["sha256"],

            "phash":
                phash
        }

    # =========================================================
    # Manifest Load / Save
    # =========================================================

    def load_manifest(self):

        if not self.manifest_path.exists():

            return []

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(
            data,
            list
        ):

            raise ValueError(
                "manifest.json 형식이 올바르지 않습니다."
            )

        return data

    def save_manifest(
        self,
        records
    ):

        with open(
            self.manifest_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                records,
                f,
                indent=4,
                ensure_ascii=False
            )

    # =========================================================
    # Duplicate Detection
    # =========================================================

    def find_duplicate(
        self,
        new_record,
        existing_records,
        phash_threshold=8
    ):

        for record in existing_records:

            # -----------------------------------------
            # Exact duplicate
            # -----------------------------------------

            if (
                new_record["sha256"]
                ==
                record["sha256"]
            ):

                return {

                    "type":
                        "exact_duplicate",

                    "matched_evidence_id":
                        record[
                            "evidence_id"
                        ],

                    "phash_distance":
                        0
                }

            # -----------------------------------------
            # Visual duplicate candidate
            # -----------------------------------------

            distance = (
                self.phash_distance(
                    new_record["phash"],
                    record["phash"]
                )
            )

            if distance <= phash_threshold:

                return {

                    "type":
                        "visual_duplicate",

                    "matched_evidence_id":
                        record[
                            "evidence_id"
                        ],

                    "phash_distance":
                        distance
                }

        return None

    # =========================================================
    # Register Evidence
    # =========================================================

    def register(
        self,
        file_path,
        evidence_id,
        source_type="original",
        phash_threshold=8
    ):

        records = self.load_manifest()

        record = self.create_record(
            file_path,
            evidence_id,
            source_type
        )

        duplicate = self.find_duplicate(
            record,
            records,
            phash_threshold
        )

        if duplicate is not None:

            return {
                "status":
                    "duplicate",

                "record":
                    record,

                "duplicate":
                    duplicate
            }

        records.append(
            record
        )

        self.save_manifest(
            records
        )

        return {
            "status":
                "registered",

            "record":
                record,

            "duplicate":
                None
        }