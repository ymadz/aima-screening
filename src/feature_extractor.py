"""Extract MobileNetV3-Small visual embeddings from tissue images."""

from os import PathLike

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import models, transforms


class MobileNetFeatureExtractor:
	"""Use the MobileNetV3-Small convolutional trunk as a feature extractor."""

	def __init__(self, pretrained: bool = True) -> None:
		weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
		base_model = models.mobilenet_v3_small(weights=weights)

		self.features = base_model.features
		self.avgpool = base_model.avgpool
		self.features.eval()
		self.avgpool.eval()

		self.transform = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
				),
			]
		)

	def extract(
		self, image_input: Image.Image | np.ndarray | str | PathLike[str]
	) -> np.ndarray:
		"""Return a 576-value embedding for a PIL image, array, or image path."""
		if isinstance(image_input, (str, PathLike)):
			try:
				image = Image.open(image_input).convert("RGB")
			except (OSError, ValueError):
				image_bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
				if image_bgr is None:
					raise ValueError(f"Could not read image: {image_input}") from None
				image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
		elif isinstance(image_input, np.ndarray):
			image = Image.fromarray(image_input).convert("RGB")
		elif isinstance(image_input, Image.Image):
			image = image_input.convert("RGB")
		else:
			raise TypeError("image_input must be a path, NumPy array, or PIL image")

		tensor = self.transform(image).unsqueeze(0)

		with torch.no_grad():
			embedding = self.avgpool(self.features(tensor))
			embedding = torch.flatten(embedding, 1).squeeze(0).cpu().numpy()

		return embedding
