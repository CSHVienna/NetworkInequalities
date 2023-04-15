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
    inequality_x, inequality_y = get_fraction_of_minority(df, x, total)
    inequity_x, inequity_y = get_gini_coefficient(df, x, total)
    return inequality_y, inequity_y


def get_fraction_of_minority(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    from netin.stats import ranking
    xs, ys = ranking.get_ranking_inequity(df, x)
    return xs, ys


def get_gini_coefficient(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    from netin.stats import ranking
    xs, ys = ranking.get_ranking_inequality(df, x, gini)
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


def fit_power_law(data: Union[np.array, Set, List], discrete: bool = True,
                  xmin: Union[None, int, float] = None, xmax: Union[None, int, float] = None, **kwargs) -> powerlaw.Fit:
    fit = powerlaw.Fit(data, discrete=discrete, xmax=xmax, xmin=xmin, **kwargs)
    return fit
