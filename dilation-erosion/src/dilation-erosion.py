#!/usr/bin/env python3

from math import floor
import sys
from pathlib import Path

import cv2
import numpy as np


class BaseImageOperation:
    def __init__(self, image_path, structuring_element):
        self.image_path = image_path
        self.image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
        (_, self.image) = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        self.width, self.height = self.image.shape
        self.structuring_element = structuring_element

        padding = floor(structuring_element.shape[1] / 2)
        # Adiciona 0 ao redor da imagem.
        self.image = np.pad(
            self.image,
            ((padding, 0), (0, padding)),
            constant_values=0,
        )

    def save(self, image, filter_name=None):
        if filter_name is None:
            filter_name = self.get_id()

        filename, extension = self.image_path.name.split(".")
        output_filename = f"{filename}-{filter_name.lower().replace(' ', '-')}.{extension}"
        output_path = Path("output", output_filename)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving output of {filter_name} filter to {output_path}...")
        cv2.imwrite(str(output_path), image)
        return output_path

    def process_pixel(self, x, y):
        (se_width, se_height) = self.structuring_element.shape

        step = floor(se_width / 2)
        sub_matrix_left = y - step
        sub_matrix_right = y + step
        sub_matrix_up = x - step
        sub_matrix_down = x + step

        se_i = 0
        for i in range(sub_matrix_up, sub_matrix_down + 1):
            if se_i >= se_height:
                continue

            se_j = 0
            for j in range(sub_matrix_left, sub_matrix_right + 1):
                if se_j >= se_width:
                    continue

                if self.pixel_should_activate(i, j, se_i, se_j):
                    return abs(255 - self.get_pixel_starting_value())
                se_j += 1
            se_i += 1
        return self.get_pixel_starting_value()

    def apply(self):
        output = np.zeros_like(self.image)
        for i in range(1, self.width):
            for j in range(1, self.height):
                output[i][j] = self.process_pixel(i, j)
        return output

    @staticmethod
    def get_pixel_starting_value():
        raise NotImplementedError

    def pixel_should_activate(self, i, j, se_i, se_j):
        raise NotImplementedError

    def get_id(self):
        raise NotImplementedError


class Dilation(BaseImageOperation):
    def get_id(self):
        return "Dilation"

    @staticmethod
    def get_pixel_starting_value():
        return 255

    def pixel_should_activate(self, i, j, se_i, se_j):
        return (
            self.structuring_element[se_i][se_j] == 255
            and self.image[i][j] != self.structuring_element[se_i][se_j]
        )


class Erosion(BaseImageOperation):
    def get_id(self):
        return "Erosion"

    @staticmethod
    def get_pixel_starting_value():
        return 0

    def pixel_should_activate(self, i, j, se_i, se_j):
        return (
            self.structuring_element[se_i][se_j] == 255
            and self.image[i][j] == self.structuring_element[se_i][se_j]
        )


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
    dilation.save(dilation.image, "Black and White")

    segmented_image = dilation.apply()
    dilation.save(segmented_image)

    erosion = Erosion(input_file, kernel)
    segmented_image = erosion.apply()
    erosion.save(segmented_image)


if __name__ == "__main__":
    main()
