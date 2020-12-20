""" Miscellaneous utilities for the project """


def merge_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    result = x.copy()
    result.update(y)
    return result
