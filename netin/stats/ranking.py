from typing import Union, Set, List

import numpy as np
import pandas as pd

from netin.utils import constants as const


### Inequity (fraction of minority in ranking) ###

def get_ranking_inequity(f_m: float, ys: np.array) -> float:
    return np.mean([efm - f_m for efm in ys if not np.isnan(efm)])


def get_ranking_inequity_class(me: float, beta: float = None) -> str:
    beta = const.INEQUITY_BETA if beta is None else beta
    label = const.INEQUITY_OVER if me > beta else const.INEQUITY_UNDER if me < -beta else const.INEQUITY_FAIR
    return label


def get_fraction_of_minority_in_ranking(df: pd.DataFrame, x: str) -> (np.ndarray, np.ndarray, float, str):
    xs = const.RANK_RANGE
    ys = []
    for rank in xs:
        column = f"{x}_rank"
        tmp = df.query(f"{column} <= @rank").copy()
        total = tmp.shape[0]
        efm = np.nan if total == 0 else tmp.query("class_label == @const.MINORITY_LABEL").shape[0] / total
        ys.append(efm)
    return xs, ys


### Inequality (Gini coefficient) ###

def get_ranking_inequality(ys: np.array) -> float:
    gini_global = ys[0]  # top-100%
    return gini_global


def get_ranking_inequality_class(gini_global: float, cuts: Set = const.INEQUALITY_CUTS) -> str:
    cuts = const.INEQUALITY_CUTS if cuts is None else cuts
    if len(cuts) != 2 or len(set(cuts)) == 1:
        raise Exception("There must be two cuts for the inequality class")

    label = const.INEQUALITY_HIGH if gini_global >= max(cuts) \
        else const.INEQUALITY_LOW if gini_global <= min(cuts) \
        else const.INEQUALITY_MODERATE
    return label


def get_gini_in_ranking(df: pd.DataFrame, x: str) -> (np.ndarray, np.ndarray, float, str):
    xs = const.RANK_RANGE
    ys = []
    for rank in xs:
        column = f"{x}_rank"
        tmp = df.query(f"{column} <= @rank").copy()
        g = gini(tmp.loc[:, x].values)
        ys.append(g)
    return xs, ys


def gini(data: np.array) -> np.array:
    """Calculate the Gini coefficient of a numpy array."""
    # https://github.com/oliviaguest/gini/blob/master/gini.py
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    X = data.flatten().astype(np.float64)
    if np.amin(X) < 0:
        # Values cannot be negative:
        X -= np.amin(X)
    # Values cannot be 0:
    X += 0.0000001
    # Values must be sorted:
    X = np.sort(X)
    # Index per array element:
    index = np.arange(1, X.shape[0] + 1)
    # Number of array elements:
    n = X.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * X)) / (n * np.sum(X))
