from typing import Union

from netin.utils import constants as const


def validate_int(value: int, minimum: int, maximum: int = None):
    if minimum in const.EMPTY and maximum in const.EMPTY:
        raise ValueError('At least one of minimum or maximum must be specified')
    if type(value) is not int:
        raise TypeError('value must be a int')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'Value is out of range.')

def validate_float(value: float, minimum: float, maximum: Union[None, float] = None, allow_none: bool = False):
    if value in const.EMPTY and allow_none:
        return True
    if value in const.EMPTY and not allow_none:
        raise ValueError('Value cannot be None')
    if type(value) is not float:
        raise TypeError('value must be a float')
    if minimum in const.EMPTY and maximum in const.EMPTY:
        raise ValueError('At least one of minimum or maximum must be specified')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'Value is out of range.')


def calibrate_homophily(h: float) -> float:
    return const.EPSILON if h == 0 else 1 - const.EPSILON if h == 1 else h
