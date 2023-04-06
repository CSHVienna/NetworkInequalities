"""

"""
from netin.utils import constants as const


def validate_int(value: int, minimum: int, maximum: int = None) -> bool:
    if minimum in const.EMPTY and maximum in const.EMPTY:
        raise ValueError('At least one of minimum or maximum must be specified')
    if type(value) is not int:
        raise TypeError('value must be a int')
    return value >= minimum if maximum is None else minimum <= value <= maximum


def validate_float(value: float, minimum: float, maximum: float, allow_none: bool=False) -> bool:
    if value in const.EMPTY and allow_none:
        return True
    if value in const.EMPTY and not allow_none:
        raise ValueError('Value cannot be None')
    if type(value) is not float:
        raise TypeError('value must be a float')
    if minimum in const.EMPTY and maximum in const.EMPTY:
        raise ValueError('At least one of minimum or maximum must be specified')
    return value >= minimum if maximum is None else minimum <= value <= maximum


def validate_homophily(h: float) -> bool:
    return const.EPSILON if h == 0 else 1 - const.EPSILON if h == 1 else h
