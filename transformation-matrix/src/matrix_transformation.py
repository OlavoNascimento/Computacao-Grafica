#!/usr/bin/env python3


from typing import Tuple
import numpy as np
from enum import Enum, auto


class ReflectionType(Enum):
    x_axis = auto()
    y_axis = auto()
    y_equal_x = auto()
    y_equal_minus_x = auto()


def translate(point: np.ndarray, Tx: int, Ty: int):
    return (point * np.array([[1, 0, Tx], [0, 1, Ty], [0, 0, 1]])).sum(axis=-1).astype(np.int32)


def scale(point: np.ndarray, Sx: int, Sy: int):
    return (point * np.array([[Sx, 0, 0], [0, Sy, 0], [0, 0, 1]])).sum(axis=-1).astype(np.int32)


def rotate(point: np.ndarray, theta: float):
    return (
        (
            point
            * np.array(
                [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            )
        )
        .sum(axis=-1)
        .astype(np.int32)
    )


def shear(point: np.ndarray, Kx: int, Ky: int):
    return (point * np.array([[1, Kx, 0], [Ky, 1, 0], [0, 0, 1]])).sum(axis=-1).astype(np.int32)


def reflect(point: np.ndarray, reflection_type: ReflectionType):
    if reflection_type == ReflectionType.x_axis:
        matrix = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    elif reflection_type == ReflectionType.y_axis:
        matrix = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif reflection_type == ReflectionType.y_equal_x:
        matrix = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    elif reflection_type == ReflectionType.y_equal_minus_x:
        matrix = [[0, -1, 0], [-1, 0, 0], [0, 0, 1]]
    else:
        raise TypeError("Invalid reflection type")
    return (point * np.array(matrix)).sum(axis=-1).astype(np.int32)


def rotate_in_relation_to_point(point: np.ndarray, reference: Tuple[int, int], theta: float):
    sin = np.sin(theta)
    cos = np.cos(theta)
    x, y, _ = point
    x_reference, y_reference = reference
    return np.array(
        [
            x_reference + cos * (x - x_reference) - (y - y_reference) * sin,
            y_reference + cos * (y - y_reference) + (x - x_reference) * sin,
            1,
        ],
        dtype=np.int32,
    )


def scale_in_relation_to_point(point: np.ndarray, reference: Tuple[int, int], Sx: int, Sy: int):
    x, y, _ = point
    x_reference, y_reference = reference
    return np.array(
        [
            x_reference + Sx * (x - x_reference),
            y_reference + Sy * (y - y_reference),
            1,
        ],
        dtype=np.int32,
    )


def main():
    point = np.array([5, 5, 1])
    reference = (1, 2)
    print(f"Translate point {point} using values 2 and 2: {translate(point, 2, 2)}")
    print(f"Scale point {point} using values 2 and 2: {scale(point, 2, 2)}")
    print(f"Rotate point {point} using angle 2: {rotate(point, 2)}")
    print(f"Shear point {point} on using values 2 and 2: {shear(point, 2, 2)}")
    print(f"Reflect point {point} on x axis: {reflect(point, ReflectionType.x_axis)}")
    print(f"Reflect point {point} on y axis: {reflect(point, ReflectionType.y_axis)}")
    print(f"Reflect point {point} on y axis = x axis: {reflect(point, ReflectionType.y_equal_x)}")
    print(
        f"Reflect point {point} on y axis = -x axis: {reflect(point, ReflectionType.y_equal_minus_x)}"
    )
    print(
        f"Scale point {point} using values 2 and 2 in relation to point {reference}: {scale_in_relation_to_point(point, reference, 2, 2)}"
    )
    print(
        f"Rotate point {point} using angle 2 in relation to point {reference}: {rotate_in_relation_to_point(point, reference, 2)}"
    )


if __name__ == "__main__":
    main()
