import numpy as np
from PIL import Image

from base_converter import ColorConverter


class CMYKConverter(ColorConverter):
    NAME = "CMYK"

    def rgb_to_cmy(self):
        cmy = self.pixels.copy()
        for i in range(self.width):
            for j in range(self.height):
                blue, green, red = self.pixels[i][j]
                cmy[i][j] = (1 - (red / 255), 1 - (green / 255), 1 - (blue / 255))
        return cmy

    @staticmethod
    def cmy_pixel_to_cmyk(cmy_pixel):
        black_key = min(cmy_pixel)
        cmyk_pixel = [0, 0, 0, black_key]
        for i in range(3):
            div = 1 - black_key
            if div == 0:
                cmyk_pixel[i] = 0
            else:
                cmyk_pixel[i] = (cmy_pixel[i] - black_key) / div
        return tuple(cmyk_pixel)

    def from_rgb(self):
        """
        Transforma uma imagem RGB em camadas CMYK.
        """
        cmy_pixels = self.rgb_to_cmy()

        c = np.zeros((self.width, self.height), dtype=float)
        m = c.copy()
        y = c.copy()
        k = c.copy()

        for i in range(self.width):
            for j in range(self.height):
                cmyk_pixel = CMYKConverter.cmy_pixel_to_cmyk(cmy_pixels[i][j])
                c[i][j], m[i][j], y[i][j], k[i][j] = cmyk_pixel

        return c, m, y, k

    def save_channels_to_file(self, c, m, y, k):
        """
        Salva as camadas de uma imagem cmyk em várias imagens em escala de cinza.
        """
        channels = (
            ("cyan", c),
            ("magenta", m),
            ("yellow", y),
            ("black", k),
        )
        super().save_channels_to_file(channels)
