#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np


class BasicImageOperation:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
        self.width, self.height = self.image.shape

    def save(self, image):
        filename, extension = self.image_path.name.split(".")
        output_filename = f"{filename}-{self.get_id().lower().replace(' ', '-')}.{extension}"
        output_path = Path("output", output_filename)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path

    def generate_histogram(self, image=None):
        if image is None:
            image = self.image

        histogram = np.zeros(256)
        for i in range(0, self.width):
            for j in range(0, self.height):
                pixel = image[i][j]
                histogram[pixel] += 1
        return histogram

    def apply(self):
        raise NotImplementedError

    def get_id(self):
        raise NotImplementedError


class SimpleSegmentation(BasicImageOperation):
    def __init__(self, image_path, initial_threshold, delta_limit):
        super().__init__(image_path)
        self.delta_limit = delta_limit
        self.initial_threshold = initial_threshold

    def get_id(self):
        return "Simple Segmentation"

    def apply(self):
        previous_threshold = None
        threshold = self.initial_threshold
        print(f"Initial threshold: {threshold}")

        _, segmented_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        iteration = 0
        while (
            previous_threshold is None or abs(threshold - previous_threshold) >= self.delta_limit
        ) and iteration < 1000:
            iteration += 1
            histogram = self.generate_histogram(segmented_image)

            group1 = 0
            group2 = 0
            div = 0
            for i in range(0, threshold):
                group1 += histogram[i] * i
                div += histogram[i]
            group1 /= div

            div = 0
            for i in range(threshold, 256):
                group2 += histogram[i] * i
                div += histogram[i]
            group2 /= div

            previous_threshold = threshold
            threshold = (group1 + group2) / 2
            print(f"Simple: Iteration {iteration} has threshold {threshold}")
            _, segmented_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)

        return segmented_image


class OtsuSegmentation(BasicImageOperation):
    def get_id(self):
        return "Otsu Segmentation"

    def apply(self):
        threshold = 1
        variance = 0

        histogram = self.generate_histogram() / (self.height * self.width)
        for i in range(1, 256):
            group1 = 0
            for j in range(0, i + 1):
                group1 += histogram[j]

            image_average = 0
            average_group1 = 0
            for j in range(0, i + 1):
                average_group1 += j * histogram[j]
                image_average += j * histogram[j]
            average_group1 /= max(group1, 1)

            group2 = 1 - group1
            average_group2 = 0
            for j in range(i + 1, 256):
                average_group2 += j * histogram[j]
                image_average += j * histogram[j]
            average_group2 /= max(group2, 1)

            new_variance = group1 * ((average_group1 - image_average) ** 2) + group2 * (
                (average_group2 - image_average) ** 2
            )
            if new_variance > variance:
                print(f"OTSU: Threshold {threshold} has larger variance than previous threshold")
                variance = new_variance
                threshold = i

        _, segmented_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        return segmented_image


def main():
    if len(sys.argv) < 2:
        print("Forneça um arquivo de entrada como argumento!", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        print(f"{input_file} não existe!", file=sys.stderr)
        sys.exit(1)

    simple_seg = SimpleSegmentation(input_file, 127, 10)
    segmented_image = simple_seg.apply()
    simple_seg.save(segmented_image)

    otsu_seg = OtsuSegmentation(input_file)
    segmented_image = otsu_seg.apply()
    otsu_seg.save(segmented_image)


if __name__ == "__main__":
    main()
