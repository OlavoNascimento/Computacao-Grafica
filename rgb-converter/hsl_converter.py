#!/usr/bin/env python3

from math import acos

import numpy as np
from PIL import Image

from base_converter import ColorConverter


class HSLConverter(ColorConverter):
    NAME = "HSL"

    @staticmethod
    def rgb_pixel_to_hsl(rgb_pixel):
        """
        Transforma um pixel RGB em um pixel HSL.
        """
        r, g, b = rgb_pixel
        r /= 255
        g /= 255
        b /= 255

        numerator = ((r - g) + (r - b)) / 2
        denominator = pow(((r - g) ** 2) + (r - b) * (g - b), 0.5)
        if denominator != 0:
            theta = numerator / denominator
        else:
            theta = 0
        theta = acos(theta)

        h = theta if b <= g else 360 - theta

        colors_sum = r + g + b
        if colors_sum != 0:
            s = 1 - (3 / colors_sum) * min(r, g, b)
        else:
            s = 0

        l = (r + g + b) / 3
        return h, s, l

    def from_rgb(self):
        """
        Transforma uma imagem RGB em camadas HSL.
        """
        h = np.zeros((self.width, self.height), dtype=float)
        s = h.copy()
        l = h.copy()

        for i in range(self.width):
            for j in range(self.height):
                hsl_pixel = HSLConverter.rgb_pixel_to_hsl(self.pixels[i][j])
                h[i][j], s[i][j], l[i][j] = hsl_pixel

        return h, s, l

    def save_channels_to_file(self, h, s, l):
        channels = (
            ("hue", h),
            ("saturation", s),
            ("lightness", l),
        )
        super().save_channels_to_file(channels)
