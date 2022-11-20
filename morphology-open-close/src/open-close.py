#!/usr/bin/env python3

from abc import abstractmethod
from math import floor
import sys
from pathlib import Path

import cv2
import numpy as np


class BasicFilter:
    def __init__(self, structuring_element):
        self.structuring_element = structuring_element

    def save(self, image, image_path: Path, filter_name=None):
        if filter_name is None:
            filter_name = self.filter_name

        output_path = Path(
            "output",
            f"{image_path.stem}-{filter_name.lower().replace(' ', '-')}{image_path.suffix}",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving output of {filter_name} filter to {output_path}...")
        cv2.imwrite(str(output_path), image)
        return output_path

    @abstractmethod
    def apply(self, image, structuring_element):
        raise NotImplementedError

    @property
    @abstractmethod
    def filter_name(self):
        raise NotImplementedError


class ImageOperation(BasicFilter):
    def __init__(self, structuring_element):
        super().__init__(structuring_element)
        (se_width, se_height) = self.structuring_element.shape
        # Distancia do pixel central até a borda do elemento estruturante.
        self.horizontal_se_distance = floor(se_width / 2)
        self.vertical_se_distance = floor(se_height / 2)

    def process_pixel(self, image, x, y):
        # Submatriz formada pelo elemento estruturante
        sub_matrix_left = y - self.horizontal_se_distance
        sub_matrix_right = y + self.horizontal_se_distance
        sub_matrix_up = x - self.vertical_se_distance
        sub_matrix_down = x + self.vertical_se_distance

        pixel_value = self.pixel_starting_value
        for i in range(sub_matrix_up, sub_matrix_down + 1):
            for j in range(sub_matrix_left, sub_matrix_right + 1):
                pixel_value = self.get_next_pixel_value(image[i][j], pixel_value)
        return pixel_value

    def pad_image(self, image):
        (se_width, se_height) = self.structuring_element.shape
        vertical_padding = floor(se_height / 2)
        horizontal_padding = floor(se_width / 2)
        # Adiciona 0 ao redor da imagem.
        padded_image = np.pad(
            image,
            ((horizontal_padding, 0), (0, vertical_padding)),
            constant_values=0,
        )
        return padded_image

    def apply(self, image):
        width, height = image.shape

        padded_image = self.pad_image(image)
        output = np.zeros_like(image)
        for i in range(1, width):
            for j in range(1, height):
                output[i][j] = self.process_pixel(padded_image, i, j)
        return output

    @abstractmethod
    def get_next_pixel_value(self, image_pixel, current_value):
        raise NotImplementedError

    @property
    @abstractmethod
    def pixel_starting_value(self):
        raise NotImplementedError


class Dilation(ImageOperation):
    filter_name = "Dilation"
    pixel_starting_value = 0

    def get_next_pixel_value(self, image_pixel, current_value):
        return max(image_pixel, current_value)


class Erosion(ImageOperation):
    filter_name = "Erosion"
    pixel_starting_value = 255

    def get_next_pixel_value(self, image_pixel, current_value):
        return min(image_pixel, current_value)


class Open(BasicFilter):
    filter_name = "Open"

    def apply(self, image):
        return Dilation(self.structuring_element).apply(
            Erosion(self.structuring_element).apply(image)
        )


class Close(BasicFilter):
    filter_name = "Close"

    def apply(self, image):
        return Erosion(self.structuring_element).apply(
            Dilation(self.structuring_element).apply(image)
        )


class MorphologicalGradient(BasicFilter):
    filter_name = "Morphological-Gradient"

    def apply(self, image):
        return cv2.absdiff(
            Dilation(self.structuring_element).apply(image),
            Erosion(self.structuring_element).apply(image),
        )


class TopHat(BasicFilter):
    filter_name = "Top-Hat"

    def apply(self, image):
        open_image = Open(self.structuring_element).apply(image)
        padded_image = np.zeros_like(open_image)
        padded_image[: image.shape[0], : image.shape[1]] = image
        return cv2.absdiff(image, open_image)


class BottomHat(BasicFilter):
    filter_name = "Bottom-Hat"

    def apply(self, image):
        closed_image = Close(self.structuring_element).apply(image)
        padded_image = np.zeros_like(closed_image)
        padded_image[: image.shape[0], : image.shape[1]] = image
        return cv2.absdiff(closed_image, image)


def main():
    if len(sys.argv) < 2:
        print("Forneça um arquivo de entrada como argumento!", file=sys.stderr)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"{image_path} não existe!", file=sys.stderr)
        sys.exit(1)

    kernel_size = int(input("Tamanho da máscara: "))
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = int(input(f"Valor da posição {i},{j}: "))
    # Considera todos os números maiores que 0 como um pixel ativo.
    kernel[kernel > 0] = 255

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    open_operation = Open(kernel)
    open_operation.save(image, image_path, "Grayscale")

    open_image = open_operation.apply(image)
    open_operation.save(open_image, image_path)

    close_operation = Close(kernel)
    closed_image = close_operation.apply(image)
    close_operation.save(closed_image, image_path)

    mg_operation = MorphologicalGradient(kernel)
    mg_image = mg_operation.apply(image)
    mg_operation.save(mg_image, image_path)

    th_operation = TopHat(kernel)
    th_image = th_operation.apply(image)
    th_operation.save(th_image, image_path)

    bt_operation = BottomHat(kernel)
    bt_image = bt_operation.apply(image)
    bt_operation.save(bt_image, image_path)


if __name__ == "__main__":
    main()
