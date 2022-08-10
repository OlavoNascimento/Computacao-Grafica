#!/usr/bin/env python3

from pathlib import Path
import sys
from cmyk_converter import CMYKConverter
from hsl_converter import HSLConverter


def main():
    if len(sys.argv) < 2:
        print("Forneça um arquivo de entrada como argumento!", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        print(f"{input_file} não existe!", file=sys.stderr)
        sys.exit(1)

    cmyk_converter = CMYKConverter(input_file)
    c, m, y, k = cmyk_converter.from_rgb()
    cmyk_converter.save_channels_to_file(c, m, y, k)

    hsl_converter = HSLConverter(input_file)
    h, s, l = hsl_converter.from_rgb()
    hsl_converter.save_channels_to_file(h, s, l)


if __name__ == "__main__":
    main()
