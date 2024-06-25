from inspect import iscoroutinefunction
from functools import wraps
from pprint import pprint
from typing import Any, Callable, Coroutine


def trace(verbose: bool = False, save_dir: str = "trace.json") -> Callable:
    def inner_trace(
        fn: Callable | Coroutine[Any, Callable, Any],
    ) -> Callable:
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            if verbose:
                pprint("=========================================")
                pprint(f"Calling {fn.__name__} with args: {args}, kwargs: {kwargs}")
            output = await fn(*args, **kwargs)
            if verbose:
                pprint(f"Called {fn.__name__} with args: {args}, kwargs: {kwargs}")
                pprint(f"Output: {output}")
                pprint("=========================================")
            return output

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if verbose:
                pprint("=========================================")
                pprint(f"Calling {fn.__name__} with args: {args}, kwargs: {kwargs}")
            output = fn(*args, **kwargs)

            if verbose:
                pprint(f"Called {fn.__name__} with args: {args}, kwargs: {kwargs}")
                pprint(f"Output: {output}")
                pprint("=========================================")
            return output

        if iscoroutinefunction(fn):
            return async_wrapper
        else:
            return wrapper

    return inner_trace
