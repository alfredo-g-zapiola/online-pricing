import io
import logging
import sys
from functools import wraps
from typing import Any


def suppress_output(func: Any) -> Any:
    @wraps(func)
    def inner(*args: Any, **kwds: Any) -> None:
        print("Running function: ", func.__name__)
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
