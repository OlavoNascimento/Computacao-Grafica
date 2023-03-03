#!/usr/bin/env python3

from abc import abstractmethod
from math import floor
import sys
from pathlib import Path

import cv2
import numpy as np


class BaseImageOperation:
    def __init__(self, image_path, structuring_element):
        self.image_path = image_path
        self.image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
        self.width, self.height = self.image.shape

        self.structuring_element = structuring_element
        (se_width, se_height) = self.structuring_element.shape
        # Distancia do pixel central até a borda do elemento estruturante.
        self.horizontal_se_distance = floor(se_width / 2)
        self.vertical_se_distance = floor(se_height / 2)

        vertical_padding = floor(se_height / 2)
        horizontal_padding = floor(se_width / 2)
        # Adiciona 0 ao redor da imagem.
        self.image = np.pad(
            self.image,
            ((horizontal_padding, 0), (0, vertical_padding)),
            constant_values=0,
        )

    def save(self, image, filter_name=None):
        if filter_name is None:
            filter_name = self.filter_name

        filename, extension = self.image_path.name.split(".")
        output_filename = f"{filename}-{filter_name.lower().replace(' ', '-')}.{extension}"
        output_path = Path("output", output_filename)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving output of {filter_name} filter to {output_path}...")
        cv2.imwrite(str(output_path), image)
        return output_path

    def process_pixel(self, x, y):
        (se_width, se_height) = self.structuring_element.shape

        # Submatriz formada pelo elemento estruturante
        horizontal_se_size = floor(se_width / 2)
        vertical_se_size = floor(se_height / 2)
        sub_matrix_left = y - horizontal_se_size
        sub_matrix_right = y + horizontal_se_size
        sub_matrix_up = x - vertical_se_size
        sub_matrix_down = x + vertical_se_size

        pixel_value = self.pixel_starting_value
        for i in range(sub_matrix_up, sub_matrix_down + 1):
            for j in range(sub_matrix_left, sub_matrix_right + 1):
                # TODO Verificar se o elemento estruturante é ativado na posição atual.
                pixel_value = self.get_next_pixel_value(self.image[i][j], pixel_value)
        return pixel_value

    def apply(self):
        output = np.zeros_like(self.image)
        for i in range(1, self.width):
            for j in range(1, self.height):
                output[i][j] = self.process_pixel(i, j)
        return output

    @abstractmethod
    def get_next_pixel_value(self, image_pixel, current_value):
        raise NotImplementedError

    @property
    @abstractmethod
    def pixel_starting_value(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def filter_name(self):
        raise NotImplementedError


class Dilation(BaseImageOperation):
    filter_name = "Dilation"
    pixel_starting_value = 0

    def get_next_pixel_value(self, image_pixel, current_value):
        return max(image_pixel, current_value)


class Erosion(BaseImageOperation):
    filter_name = "Erosion"
    pixel_starting_value = 255

    def get_next_pixel_value(self, image_pixel, current_value):
        return min(image_pixel, current_value)


def main():
    if len(sys.argv) < 2:
        print("Forneça um arquivo de entrada como argumento!", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        print(f"{input_file} não existe!", file=sys.stderr)
        sys.exit(1)

    kernel_size = int(input("Tamanho da máscara: "))
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = int(input(f"Valor da posição {i},{j}: "))
    # Considera todos os números maiores que 0 como um pixel ativo.
    kernel[kernel > 0] = 255

    dilation = Dilation(input_file, kernel)
    dilation.save(dilation.image, "Grayscale")

    segmented_image = dilation.apply()
    dilation.save(segmented_image)

    erosion = Erosion(input_file, kernel)
    segmented_image = erosion.apply()
    erosion.save(segmented_image)


if __name__ == "__main__":
    main()
