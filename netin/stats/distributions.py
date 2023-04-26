from typing import Union, Set, List, Tuple

import numpy as np
import pandas as pd
import powerlaw


def get_pdf(df: pd.DataFrame, x: str, total: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the probability density of the input data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the data.
    x : str
        The column name of the data.
    total : float
        The total amount by which to normalize the data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays holding the x values and y values (their probability).
    """
    values = df.groupby(x).size()
    xs = values.index.values
    ys = values.values / total
    return xs, ys


def get_cdf(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    """Computes the cumulative distribution CDF of the input data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the data.
    x : str
        The column name of the data.
    total : float
        The total amount by which to normalize the data. (not used here)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays holding the x values and the y values (CDF)
    """
    xs = np.sort(df[x].values)
    n = xs.size
    ys = np.arange(1, n + 1) / n
    return xs, ys


def get_ccdf(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    """Computes the complementary cumulative distribution CCDF of the input data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the data.
    x : str
        The column name of the data.
    total : float
        The total amount by which to normalize the data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays holding the x values and the y values (CCDF)
    """
    xs, ys = get_cdf(df, x, total)
    return xs, 1 - ys


def get_disparity(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    """Computes the disparity of the input data given by the column `x`.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains the data.

    x: str
        The column name of the data.

    total: float
        The total amount by which to normalize the data. (not used here)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays holding the x values (ranking) and the y values (disparity)
    """
    from netin.stats import ranking
    from netin.utils import constants as const

    gx, gy = get_gini_coefficient(df, x, total)
    fx, fy = get_fraction_of_minority(df, x, total)
    f_m = df.query("class_label == @const.MINORITY_LABEL").shape[0] / df.shape[0]

    inequality_y = ranking.get_ranking_inequality(gy)
    inequity_x = ranking.get_ranking_inequity(f_m, fy)
    return inequity_x, inequality_y


def get_fraction_of_minority(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    """Computes the fraction of minority in each top-k rank.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains the data.

    x: str
        The column name of the data.

    total: float
        The total amount by which to normalize the data. (not used here)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays holding the x values (ranking) and the y values (fraction of minority)
    """
    from netin.stats import ranking
    xs, ys = ranking.get_fraction_of_minority_in_ranking(df, x)
    return xs, ys


def get_gini_coefficient(df: pd.DataFrame, x: str, total: float = None) -> (np.ndarray, np.ndarray):
    """Computes the Gini coefficient of the distribution in each top-k rank.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains the data.

    x: str
        The column name of the data.

    total: float
        The total amount by which to normalize the data. (not used here)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays holding the x values (ranking) and the y values (Gini coefficient)
    """
    from netin.stats import ranking
    xs, ys = ranking.get_gini_in_ranking(df, x)
    return xs, ys


def fit_power_law(data: Union[np.array, Set, List], discrete: bool = True,
                  xmin: Union[None, int, float] = None, xmax: Union[None, int, float] = None, **kwargs) -> powerlaw.Fit:
    """Fits a power-law of a given distribution.

    Parameters
    ----------
    data: Union[np.array, Set, List]
        The data to fit.

    discrete: bool
        Whether the data is discrete or not.

    xmin: Union[None, int, float]
        The minimum value of the data.

    xmax: Union[None, int, float]
        The maximum value of the data.

    kwargs: dict
        Additional arguments to pass to the powerlaw.Fit constructor.

    Returns
    -------
    powerlaw.Fit
        The fitted power-law.
    """
    fit = powerlaw.Fit(data, discrete=discrete, xmax=xmax, xmin=xmin, **kwargs)
    return fit
