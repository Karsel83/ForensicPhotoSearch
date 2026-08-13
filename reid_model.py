import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DEEP_PERSON_REID_PATH = os.path.join(
    PROJECT_ROOT,
    "deep-person-reid"
)

if DEEP_PERSON_REID_PATH not in sys.path:
    sys.path.insert(0, DEEP_PERSON_REID_PATH)

import torch
from torchreid.utils import FeatureExtractor


class PersonReID:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[*] Re-ID device: {self.device}")

        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device=self.device
        )

        print("[*] OSNet loaded")

    def extract(self, image_path):

        features = self.extractor(
            [image_path]
        )

        embedding = (
            features[0]
            .detach()
            .cpu()
            .numpy()
        )

        return embedding
