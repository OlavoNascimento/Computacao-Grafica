#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
from pathlib import Path
from PIL import Image

import cv2
import numpy as np


class Histogram:
    def __init__(self, image_path):
        self.image_path = image_path
        self.pixels = cv2.imread(str(image_path))
        self.width, self.height, _ = self.pixels.shape

    @staticmethod
    def display_histogram(title, *hists):
        x_axis = range(255)
        length = len(hists)
        fig, plots = plt.subplots(length)
        fig.suptitle(title)
        for i, (hist, subtitle) in enumerate(hists):
            plots[i].plot(x_axis, hist)
            plots[i].set_title(subtitle)
        plt.show()

    def split_into_channels(self):
        r = np.zeros((self.width, self.height), dtype=np.uint8)
        g = r.copy().astype(np.uint8)
        b = r.copy().astype(np.uint8)
        for i in range(self.width):
            for j in range(self.height):
                b[i][j], g[i][j], r[i][j] = self.pixels[i][j]
        return r, g, b

    def create_histogram(self):
        r, g, b = self.split_into_channels()
        r_hist = np.zeros(255, dtype=np.uint64)
        g_hist = r_hist.copy().astype(np.uint64)
        b_hist = r_hist.copy().astype(np.uint64)

        for channel, hist in [(r, r_hist), (g, g_hist), (b, b_hist)]:
            for i in range(self.width):
                for j in range(self.height):
                    value = channel[i][j]
                    hist[value] += 1

        # self.display_histogram("Histogramas", (r_hist, "Red"), (g_hist, "Green"), (b_hist, "Blue"))
        return r_hist, g_hist, b_hist

    def normalize_histogram(self):
        r_hist, g_hist, b_hist = self.create_histogram()
        total_pixels = self.width * self.height

        r_acc = np.zeros(255, dtype=np.float64)
        g_acc = r_acc.copy().astype(np.float64)
        b_acc = r_acc.copy().astype(np.float64)

        for hist, acc in [(r_hist, r_acc), (g_hist, g_acc), (b_hist, b_acc)]:
            for i in range(255):
                acc[i] = hist[i] / total_pixels

        # self.display_histogram(
        #     "Histogramas normalizados", (r_acc, "Red"), (g_acc, "Green"), (b_acc, "Blue")
        # )
        return r_acc, g_acc, b_acc

    def accumulate_histogram(self):
        r_hist, g_hist, b_hist = self.normalize_histogram()

        r_acc = np.zeros(255, dtype=np.float64)
        g_acc = r_acc.copy().astype(np.float64)
        b_acc = r_acc.copy().astype(np.float64)

        for hist, acc in [(r_hist, r_acc), (g_hist, g_acc), (b_hist, b_acc)]:
            for i in range(255):
                previous_index = max(i - 1, 0)
                acc[i] = hist[i] + acc[previous_index]

        # self.display_histogram(
        #     "Histogramas acumulados", (r_acc, "Red"), (g_acc, "Green"), (b_acc, "Blue")
        # )
        return r_acc, g_acc, b_acc

    def equalize_image(self):
        r_acc, g_acc, b_acc = self.accumulate_histogram()

        new_image = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        for i in range(self.width):
            for j in range(self.height):
                b, g, r = self.pixels[i][j]
                new_pixel = (round(255 * r_acc[r]), round(255 * g_acc[g]), round(255 * b_acc[b]))
                new_image[i][j] = new_pixel

        output_path = Path("images", f"{self.image_path.stem}-equalized.jpg")
        output_path.parent.mkdir(exist_ok=True)
        with Image.fromarray(new_image, "RGB") as image:
            image.save(output_path)


def main():
    if len(sys.argv) < 2:
        print("Forneça um arquivo de entrada como argumento!", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        print(f"{input_file} não existe!", file=sys.stderr)
        sys.exit(1)

    hist = Histogram(input_file)
    hist.equalize_image()


if __name__ == "__main__":
    main()
