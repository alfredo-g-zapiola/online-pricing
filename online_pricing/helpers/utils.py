import io
import itertools
import logging
import sys
import threading
import time
from functools import wraps
from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def animate(prefix: str, stop: Callable[[], bool], stdout: Any) -> None:
    while True:
        for c in itertools.cycle(["|", "/", "-", "\\"]):
            old_stdout = sys.stdout
            sys.stdout = stdout
            print(prefix + c, end="\r")
            sys.stdout = old_stdout
            time.sleep(0.2)
            if stop():
                return


def suppress_output(func: Any) -> Any:
    @wraps(func)
    def inner(*args: Any, **kwds: Any) -> None:

        # Disable logging
        logging.disable(logging.CRITICAL)
        # Disable stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        stop_threads = False
        t = threading.Thread(target=animate, args=(f"Running function: {func.__name__} ", lambda: stop_threads, old_stdout))
        t.start()

        func(*args, **kwds)
        # Restore stdout
        sys.stdout = old_stdout
        # Restore logging
        logging.disable(logging.NOTSET)

        stop_threads = True
        t.join()

    return inner


def flatten(my_list: list[list[Any]] | npt.NDArray[np.float64]) -> list[Any]:
    return [item for sublist in my_list for item in sublist]


def sum_by_element(array_1: Any, array_2: Any, difference: bool = False) -> Any:
    """
    Sum lists - or matrices - by element.

    :param array_1: list or matrix to sum.
    :param array_2: list or matrix to sum.
    :param difference: if True, return the difference between the two arrays.
    :return: list or matrix with the sum of the two arrays.
    """
    if type(array_1) is not type(array_2):
        raise TypeError(f"Arrays must be of the same type, got {type(array_1)} and {type(array_2)}")

    if isinstance(array_1[0], list):
        return [sum_by_element(a1, a2) for a1, a2 in zip(array_1, array_2)]

    if difference:
        return [a1 - a2 for a1, a2 in zip(array_1, array_2)]

    return [sum(items) for items in zip(array_1, array_2)]


def print_matrix(matrix: list[list[float | int]], indexes: bool = False) -> None:
    if indexes:
        indices = ["-"] + [str(i) for i in range(1, len(matrix[0]) + 1)]
        new_matrix = [[str(matrix[j][i]) for i in range(len(matrix[0]))] for j in range(len(matrix))]
        new_matrix.insert(0, indices)
        for idx in range(1, len(new_matrix)):
            new_matrix[idx].insert(0, str(idx))

        print("\n".join(["".join([f"{item} " for item in row]) for row in new_matrix]))

    elif type(matrix[0][0]) == int:
        print("\n".join(["".join([f"{item} " for item in row]) for row in matrix]))
    else:
        print("\n".join(["".join([f"{item:.2f} " for item in row]) for row in matrix]))


def int_to_features(num: int) -> list[int]:
    return [int(feature) for feature in f"{num:02b}"]


def mean(my_list: Any) -> float:
    return sum(my_list) / len(my_list)
