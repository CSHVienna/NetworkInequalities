from typing import Union, Set, List

import numpy as np
import pandas as pd
from numpy import ndarray

from netin.utils import constants as const


def get_ranking_me(f_m: float, ys: np.array) -> float:
    return np.mean([efm - f_m for efm in ys if not np.isnan(efm)])


def get_ranking_inequity_class(me: float, beta: float = None) -> str:
    beta = const.INEQUITY_BETA if beta is None else beta
    label = const.INEQUITY_OVER if me > beta else const.INEQUITY_UNDER if me < -beta else const.INEQUITY_FAIR
    return label


def get_ranking_inequity(df: pd.DataFrame, x: str) -> (np.ndarray, np.ndarray, float, str):
    xs = const.RANK_RANGE
    ys = []
    for rank in xs:
        total = df.query(f"{x}_rank <= @rank").shape[0]
        efm = np.nan if total == 0 else df.query(f"{x}_rank <= @rank and class_label == @const.MINORITY_LABEL").shape[
                                            0] / total
        ys.append(efm)
    return xs, ys


def get_ranking_inequality(df: pd.DataFrame, x: str, inequality_fnc: callable) -> (np.ndarray, np.ndarray, float, str):
    xs = const.RANK_RANGE
    ys = []
    for rank in xs:
        tmp = df.query(f"{x}_rank <= @rank").copy()
        g = inequality_fnc(tmp.loc[:, x].values)
        ys.append(g)
    return xs, ys


def get_ranking_inequality_class(gini: float, cuts: Set = const.INEQUALITY_CUTS) -> str:
    cuts = const.INEQUALITY_CUTS if cuts is None else cuts
    if len(cuts) != 2 or len(set(cuts)) == 1:
        raise Exception("There must be two cuts for the inequality class")

    label = const.INEQUALITY_HIGH if gini >= max(cuts) \
        else const.INEQUALITY_LOW if gini <= min(cuts) \
        else const.INEQUALITY_MODERATE
    return label
