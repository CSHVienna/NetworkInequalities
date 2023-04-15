from typing import Union, Set, List

import numpy as np
import pandas as pd
import powerlaw


def get_pdf(df: pd.DataFrame, x: str, total: float) -> (np.ndarray, np.ndarray):
    values = df.groupby(x).size()
    xs = values.index.values
    ys = values.values / total
    return xs, ys


def get_cdf(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    xs = np.sort(df[x].values)
    n = xs.size
    ys = np.arange(1, n + 1) / n
    return xs, ys


def get_ccdf(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    xs, ys = get_cdf(df, x, total)
    return xs, 1 - ys


def get_disparity(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    from netin.stats import ranking
    from netin.utils import constants as const

    gx, gy = get_gini_coefficient(df, x, total)
    fx, fy = get_fraction_of_minority(df, x, total)
    f_m = df.query("class_label == @const.MINORITY_LABEL").shape[0] / df.shape[0]

    inequality_y = ranking.get_ranking_inequality(gy)
    inequity_x = ranking.get_ranking_inequity(f_m, fy)
    return inequity_x, inequality_y


def get_fraction_of_minority(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    from netin.stats import ranking
    xs, ys = ranking.get_fraction_of_minority_in_ranking(df, x)
    return xs, ys


def get_gini_coefficient(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    from netin.stats import ranking
    xs, ys = ranking.get_gini_in_ranking(df, x)
    return xs, ys


def fit_power_law(data: Union[np.array, Set, List], discrete: bool = True,
                  xmin: Union[None, int, float] = None, xmax: Union[None, int, float] = None, **kwargs) -> powerlaw.Fit:
    fit = powerlaw.Fit(data, discrete=discrete, xmax=xmax, xmin=xmin, **kwargs)
    return fit
