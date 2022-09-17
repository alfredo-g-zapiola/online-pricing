import io
import logging
import sys
from functools import wraps
from typing import Any


def suppress_output(func: Any) -> Any:
    @wraps(func)
    def inner(*args: Any, **kwds: Any) -> None:
        print(f"Running function: {func.__name__} \n")
        # Disable logging
        logging.disable(logging.CRITICAL)
        # Disable stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        func(*args, **kwds)

        # Restore stdout
        sys.stdout = old_stdout
        # Restore logging
        logging.disable(logging.NOTSET)

    return inner


def flatten(my_list: list[list[Any]]) -> list[Any]:
    return [item for sublist in my_list for item in sublist]
