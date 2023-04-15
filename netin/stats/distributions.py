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


def fit_power_law(data: Union[np.array, Set, List], discrete: bool = True,
                  xmin: Union[None, int, float] = None, xmax: Union[None, int, float] = None, **kwargs) -> powerlaw.Fit:
    fit = powerlaw.Fit(data, discrete=discrete, xmax=xmax, xmin=xmin, **kwargs)
    return fit
