from typing import Union, Set, List, Tuple

import numpy as np
import pandas as pd

from netin.utils import constants as const


### Inequity (fraction of minority in ranking) ###

def get_ranking_inequity(f_m: float, ys: np.array) -> float:
    """Computes ME: mean error distance between the fraction of minority in each top-k rank `f_m^k` and
    the fraction of minority of the entire graph `f_m`. ME is the ranking inequity of the rank.

    Parameters
    ----------
    f_m: float
        The fraction of minority in the entire graph.

    ys: np.array
        The fraction of minority in each top-k rank.

    Returns
    -------
    me: float
        The ranking inequity of the rank.
    """
    me = np.mean([efm - f_m for efm in ys if not np.isnan(efm)])
    return me


def get_ranking_inequity_class(me: float, beta: float = None) -> str:
    """
    Infers the inequity class (label) given the inequity measure (ME).

    Parameters
    ----------
    me: float
        The inequity measure (ME).

    beta: float
        The threshold to determine the inequity class.

    Returns
    -------
    label: str
        The inequity class label (i.e., fair, over-represented, under-represented).

    Notes
    -----
    See :func:`get_ranking_inequity` for more details on `me`.

    By default, `beta=0.05`, see [Espin-Noboa2022].
    """
    beta = const.INEQUITY_BETA if beta is None else beta
    label = const.INEQUITY_OVER if me > beta else const.INEQUITY_UNDER if me < -beta else const.INEQUITY_FAIR
    return label


def get_fraction_of_minority_in_ranking(df: pd.DataFrame, x: str) -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[list, list]]:
    """
    Computes the fraction of minority in each top-k rank.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains the data.

    x: str
        The column name of the data.

    Returns
    -------
    xs: np.ndarray
        The x values (ranking).

    ys: np.ndarray
        The y values (fraction of minority).
    """
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
    """
    Returns the Gini coefficient of the entire distribution (at op-100%).

    Parameters
    ----------
    ys: np.array
        The y values (Gini coefficients in each top-k rank).

    Returns
    -------
    float
        The Gini coefficient of the entire distribution (at op-100%).
    """
    gini_global = ys[0]  # top-100%
    return gini_global


def get_ranking_inequality_class(gini_global: float, cuts: Set[float] = const.INEQUALITY_CUTS) -> str:
    """
    Infers the inequality class label given the Gini coefficient of the entire distribution.

    Parameters
    ----------
    gini_global: float
        The Gini coefficient of the entire distribution.

    cuts: Set[float]
        The cuts to determine the inequality class.

    Returns
    -------
    label: str
        The inequality class label (i.e., equality, moderate, skewed)

    Notes
    -----
    By default, `cuts={0.3, 0.6}`, see [Espin-Noboa2022].
    """
    cuts = const.INEQUALITY_CUTS if cuts is None else cuts
    if len(cuts) != 2 or len(set(cuts)) == 1:
        raise Exception("There must be two cuts for the inequality class")

    label = const.INEQUALITY_HIGH if gini_global >= max(cuts) \
        else const.INEQUALITY_LOW if gini_global <= min(cuts) \
        else const.INEQUALITY_MODERATE
    return label


def get_gini_in_ranking(df: pd.DataFrame, x: str) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[list, list]]:
    """
    Computes the Gini coefficient of a distribution `df[x]` in each top-k rank.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe that contains the data.

    x: str
        The column name of the data.

    Returns
    -------
    xs: np.ndarray
        The x values (ranking).
    ys: np.ndarray
        The y values (Gini coefficients).
    """
    xs = const.RANK_RANGE
    ys = []
    for rank in xs:
        column = f"{x}_rank"
        tmp = df.query(f"{column} <= @rank").copy()
        g = gini(tmp.loc[:, x].values)
        ys.append(g)
    return xs, ys


def gini(data: np.array) -> float:
    """
    Calculates the Gini coefficient of a distribution.

    Parameters
    ----------
    data: np.array
        The data.

    Returns
    -------
    float
        The Gini coefficient of the distribution.

    References
    ----------
    `Gini coefficient <http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm>`_
    `Implementation <https://github.com/oliviaguest/gini/blob/master/gini.py>`_
    """
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
