#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt


def linha_decrescente(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    increment = 1
    if dy < 0:
        increment = -1
        dy = -dy

    p = 2 * dy - dx
    p2 = 2 * dy
    twice_xy = 2 * (dy - dx)

    points = []
    y = y0
    for x in range(x0, x1):
        points.append((x, y))
        if p > 0:
            y = y + increment
            p = p + twice_xy
        else:
            p = p + p2
    return points


def linha_crescente(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    increment = 1
    if dx < 0:
        increment = -1
        dx = -dx

    p = 2 * dx - dy
    p2 = 2 * dx
    twice_xy = 2 * (dx - dy)

    points = []
    x = x0
    for y in range(y0, y1):
        points.append((x, y))
        if p > 0:
            x = x + increment
            p = p + twice_xy
        else:
            p = p + p2
    return points


def bresenham(x0, y0, x1, y1):
    dist_y = abs(y1 - y0)
    dist_x = abs(x1 - x0)
    start = (x0, y0)
    end = (x1, y1)
    if y0 > y1:
        start = (x1, y1)
        end = (x0, y0)

    if dist_y >= dist_x:
        points = linha_crescente(start[0], start[1], end[0], end[1])
    else:
        points = linha_decrescente(start[0], start[1], end[0], end[1])

    points.insert(0, (x0, y0))
    points.append((x1, y1))
    return points


def circuferencia(centro_x, centro_y, r):
    x = r
    y = 0

    points = []
    points.append((x + centro_x, y + centro_y))
    if r > 0:
        for (x_reflection, y_reflection) in [(x, -y), (y, x), (-y, x)]:
            points.append((x_reflection + centro_x, y_reflection + centro_y))

    p = 1 - r
    while x > y:
        y += 1
        if p <= 0:
            p = p + 2 * y + 1
        else:
            x -= 1
            p = p + 2 * y - 2 * x + 1
        if x < y:
            break

        for (x_reflection, y_reflection) in [(x, y), (-x, y), (x, -y), (-x, -y)]:
            points.append((x_reflection + centro_x, y_reflection + centro_y))

        if x != y:
            for (x_reflection, y_reflection) in [(y, x), (-y, x), (y, -x), (-y, -x)]:
                points.append((x_reflection + centro_x, y_reflection + centro_y))
    return points


def main():
    x1, y1 = (1, 1)
    x2, y2 = (-15, -25)
    points = bresenham(x1, y1, x2, y2)
    plt.scatter([point[0] for point in points], [point[1] for point in points])
    plt.show()

    points = circuferencia(x1, y1, 10)
    plt.scatter([point[0] for point in points], [point[1] for point in points])
    plt.show()


if __name__ == "__main__":
    main()
