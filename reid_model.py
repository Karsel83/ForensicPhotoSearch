import torch
from torchreid.utils import FeatureExtractor


class PersonReID:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[*] Re-ID device: {self.device}")

        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device=self.device
        )

        print("[*] OSNet loaded")

    def extract(self, image_path):

        features = self.extractor([image_path]) 

        embedding = features[0].detach().cpu().numpy()# 해당사람의 이미지 특징 벡터

        return embedding