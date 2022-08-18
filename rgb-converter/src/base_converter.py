from abc import ABCMeta, abstractmethod
from pathlib import Path
from PIL import Image
import cv2
import numpy as np


class ColorConverter(metaclass=ABCMeta):
    NAME = "BASE"

    def __init__(self, image_path):
        self.image_path = image_path
        self.pixels = cv2.imread(str(image_path)).astype(float)
        self.width, self.height, _ = self.pixels.shape

    def save_channels_to_file(self, channels):
        """
        Salva várias camadas em imagens em escala de cinza.
        """
        for name, channel in channels:
            values = (channel.copy() * 255).astype(np.uint8)
            # Salva a camada em uma imagem em escala de cinza.
            output_path = Path(self.NAME, f"{self.image_path.stem}-{name}.jpg")
            output_path.parent.mkdir(exist_ok=True)
            with Image.fromarray(values, "L") as image:
                image.save(output_path)

    @abstractmethod
    def from_rgb(self):
        raise NotImplementedError("O método from_rgb() deve ser implementado")
