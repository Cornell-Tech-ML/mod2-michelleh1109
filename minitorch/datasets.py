import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates N random 2D points in the unit square [0, 1) x [0, 1).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of tuples representing random 2D points.

    """ """Generates N random 2D points in the unit square [0, 1) x [0, 1).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of tuples representing random 2D points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """A simple data structure representing a graph with N points and their corresponding labels.

    Attributes
    ----------
        N (int): The number of points.
        X (List[Tuple[float, float]]): List of 2D points.
        y (List[int]): List of labels for each point.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset where points are labeled based on whether the sum of their coordinates
    is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset where points are labeled based on whether their x_1 coordinate is
    less than 0.2 or greater than 0.8.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset where points are labeled based on an XOR pattern. Points are labeled
    as 1 if x_1 and x_2 are on opposite sides of 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset where points are labeled based on whether they lie inside or outside
    a circle centered at (0.5, 0.5) with a radius squared of 0.1.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a spiral dataset where points follow two spirals starting from the center.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and binary labels, with points following a spiral pattern.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
