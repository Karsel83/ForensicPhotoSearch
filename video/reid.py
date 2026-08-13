import torch
import torchreid
import cv2
import numpy as np


class PersonReID:

    def __init__(self):

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )

        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

    def extract(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 128))

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        tensor = torch.from_numpy(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        with torch.no_grad():

            feature = self.model(tensor)

        feature = feature.cpu().numpy()[0]

        feature = feature / (
            np.linalg.norm(feature) + 1e-12
        )

        return feature