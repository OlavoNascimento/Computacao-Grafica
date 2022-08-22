#!/usr/bin/env python3

import sys
from math import floor
from pathlib import Path

import cv2
import numpy as np


class ConvolutionFilter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        self.width, self.height, _ = self.image.shape

    def process_pixel(self, x, y, image, mask):
        (mask_width, _) = mask.shape
        step = floor(mask_width / 2)
        b_median = 0
        g_median = 0
        r_median = 0

        sub_matrix_left = y - step
        sub_matrix_right = y + step
        sub_matrix_up = x - step
        sub_matrix_down = x + step

        for i in range(sub_matrix_up, sub_matrix_down + 1):
            for j in range(sub_matrix_left, sub_matrix_right + 1):
                mask_x = i - sub_matrix_up
                mask_y = j - sub_matrix_left
                weight = mask[mask_x][mask_y]

                b, g, r = image[i][j]
                b_median += b * weight
                g_median += g * weight
                r_median += r * weight
        return b_median, g_median, r_median

    def apply(self, mask):
        (height, _) = mask.shape
        padding = floor(height / 2)

        # Adiciona 1 ao redor da imagem.
        image = np.pad(
            self.image,
            ((padding, 0), (0, padding), (0, 0)),
            constant_values=1,
        ).astype(np.float64)

        output = np.zeros(image.shape, dtype=np.float64)
        for i in range(padding, self.height):
            for j in range(padding, self.width):
                median_pixel = self.process_pixel(i, j, image, mask)
                output[i][j] = median_pixel
        return output

    def save(self, image, filter_name):
        filename, extension = self.image_path.name.split(".")
        output_filename = f"{filename}-{filter_name.replace(' ', '-')}.{extension}"
        output_path = Path("output", output_filename)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path


def main():
    if len(sys.argv) < 2:
        print("Forneça um arquivo de entrada como argumento!", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        print(f"{input_file} não existe!", file=sys.stderr)
        sys.exit(1)

    mask_size = int(input("Tamanho de máscara desejada para o filtro de média: "))

    conv_filter = ConvolutionFilter(input_file)

    weights = {
        "média": [[1 / (mask_size**2)] * mask_size] * mask_size,
        "norte": [[1, 1, 1], [1, -2, 1], [-1, -1, -1]],
        "sul": [[-1, -1, -1], [1, -2, 1], [1, 1, 1]],
        "leste": [[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]],
        "oeste": [[1, 1, -1], [1, -2, -1], [1, 1, -1]],
        "nordeste": [[1, 1, 1], [-1, -2, 1], [-1, -1, 1]],
        "noroeste": [[1, 1, 1], [1, -2, -1], [1, -1, -1]],
        "sudeste": [[-1, -1, 1], [-1, -2, 1], [1, 1, 1]],
        "sudoeste": [[1, -1, -1], [1, -2, -1], [1, 1, 1]],
    }
    for filter_name, weight in weights.items():
        weighted_image = conv_filter.apply(np.array(weight, dtype=np.float64))
        output_path = conv_filter.save(weighted_image, filter_name)
        print(
            f"Aplicando filtro {filter_name} em {input_file} e salvando resultado em {output_path}"
        )

    weights = {
        "sobel horizontal": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "sobel vertical": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    }
    sobel_results = []
    for filter_name, weight in weights.items():
        conv_filter = ConvolutionFilter(input_file)
        weighted_image = conv_filter.apply(np.array(weight, dtype=np.float64))
        output_path = conv_filter.save(weighted_image, filter_name)
        print(f"Aplicando filtro sobel em {input_file} e salvando resultado em {output_path}")
        sobel_results.append(weighted_image)
    output_path = conv_filter.save(sobel_results[0] + sobel_results[1], "sobel soma")
    print(f"Aplicando filtro sobel soma em {input_file} e salvando resultado em {output_path}")


if __name__ == "__main__":
    main()
