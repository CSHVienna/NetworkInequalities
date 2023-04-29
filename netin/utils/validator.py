from typing import Union
import warnings

import networkx as nx

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


def validate_values(value: object, values: list):
    if value not in values:
        raise ValueError(f'Value must be one of {values}')


def calibrate_null_probabilities(p: float) -> float:
    return const.EPSILON if p == 0 else 1 - const.EPSILON if p == 1 else p


def validate_graph_metadata(g: Union[nx.Graph, nx.DiGraph]):
    err = []
    for gkey in ['class_attribute', 'class_values', 'class_labels']:
        if gkey not in g.graph:
            err.append(gkey)
    if len(err) > 0:
        raise ValueError(f'Graph must have these attributes: "{", ".join(err)}".')

    nkey = g.graph['class_attribute']
    for n, obj in g.nodes(data=True):
        if nkey not in obj:
            raise ValueError(f'Nodes must have a "{nkey}" attribute')
        break


def validate_more_than_one(iterable):
    if len(iterable) < 2:
        raise ValueError('At least two elements are required')


def ignore_params(params: list, **kwargs) -> dict:
    tmp = set()
    for param in params:
        p = kwargs.pop(param, None)
        if p is not None:
            tmp.add(param)
    if len(tmp) > 0:
        warnings.warn(f"These parameters are ignored: {', '.join(tmp)}")

    return kwargs
