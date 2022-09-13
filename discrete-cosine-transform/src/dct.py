#!/usr/bin/env python3

import sys
from math import pi, cos, sqrt
from pathlib import Path

import cv2
import numpy as np


class CosineFilter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))

    @staticmethod
    def dct_cosine(x, i, n):
        return cos(((2.0 * x + 1) * i * np.pi) / (2 * n))

    @staticmethod
    def alpha(u, n):
        div = 2
        if u == 0:
            div = 1
        return sqrt(div / n)

    def dct(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        (height, width) = image.shape
        discrete_cosine = np.zeros_like(image, dtype=np.float64)

        cosines = np.zeros((height, width), dtype=np.float64)
        for i in range(height):
            for j in range(width):
                cosines[i][j] = self.dct_cosine(i, j, width)

        for i in range(height):
            for j in range(width):
                cos_sum = 0

                for x in range(height):
                    for y in range(width):
                        cos_sum += image[x][y] * cosines[x][i] * cosines[j][y]

                discrete_cosine[i][j] = cos_sum * self.alpha(i, height) * self.alpha(j, width)

        return discrete_cosine

    def idct(self, image):
        (height, width) = image.shape
        inverse_discrete_cosine = np.zeros_like(image, dtype=np.float64)

        cosines = np.zeros((height, width), dtype=np.float64)
        for i in range(height):
            for j in range(width):
                cosines[i][j] = self.dct_cosine(i, j, width)

        for i in range(height):
            for j in range(width):
                cos_sum = 0

                for x in range(height):
                    alpha_x = self.alpha(x, width)
                    for y in range(width):
                        cos_sum += (
                            alpha_x
                            * image[x][y]
                            * self.alpha(y, width)
                            * cosines[x][i]
                            * cosines[y][j]
                        )
                inverse_discrete_cosine[i][j] = cos_sum
        return inverse_discrete_cosine

    def filter_pass(self, condition):
        output_image = np.zeros_like(self.image, dtype=np.uint8)
        (height, width, _) = self.image.shape
        origin_x = width / 2
        origin_y = height / 2

        for i in range(height):
            for j in range(width):
                dist = sqrt(((i - origin_y) ** 2) + ((j - origin_x) ** 2))
                if condition(dist):
                    color = 255
                else:
                    color = 0
                output_image[i][j] = np.full(3, color)

        return output_image

    def low_pass(self, max_dist):
        return self.filter_pass(lambda x: x <= max_dist)

    def high_pass(self, max_dist):
        return self.filter_pass(lambda x: x > max_dist)

    def save(self, image, filter_name):
        filename, extension = self.image_path.name.split(".")
        output_filename = f"{filename}_{filter_name.replace(' ', '_').lower()}.{extension}"
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

    max_dist = 20

    cosine = CosineFilter(input_file)
    outputs = {}
    filters = {
        "low pass": lambda: cosine.low_pass(max_dist),
        "high pass": lambda: cosine.high_pass(max_dist),
        "DCT": cosine.dct,
        "IDCT": lambda: cosine.idct(outputs["DCT"]),
    }
    for name, func in filters.items():
        dct_output_image = func()
        output_path = cosine.save(dct_output_image, name)
        outputs[name] = dct_output_image
        print(f"Aplicando filtro {name} em {input_file} e salvando resultado em {output_path}")

    image_with_noise = cosine.image.copy()
    (height, width, _) = image_with_noise.shape
    for i in range(height):
        for j in range(width):
            if j % 2 == 0:
                image_with_noise[i][j] -= max(int(i * 1.5 - j), 255)

    noise_filters = {"low pass noise": filters["low pass"], "high pass noise": filters["high pass"]}
    for name, func in noise_filters.items():
        dct_output_image = func()
        output_path = cosine.save(dct_output_image, name)
        outputs[name] = dct_output_image
        print(f"Aplicando filtro {name} em {input_file} e salvando resultado em {output_path}")

    # Ruído no domínio de frequência.
    dct_output_image = cosine.dct()
    (height, width) = dct_output_image.shape
    for i in range(height):
        for j in range(width):
            if j % 2 == 0:
                dct_output_image[i][j] -= max(int(i * 1.5 - j), 255)
    # Converte para o domínio espacial.
    output_path = cosine.save(cosine.idct(dct_output_image), "IDCT ruído")
    print(f"Aplicando filtro IDCT com ruído em {input_file} e salvando resultado em {output_path}")


if __name__ == "__main__":
    main()
