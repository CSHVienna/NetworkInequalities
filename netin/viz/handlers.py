import logging
from typing import Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from netin.utils import constants as const
from netin.viz.constants import *
from netin.generators.graph import Graph
from netin.stats.distributions import fit_power_law

def _get_edge_color(s: int, t: int, g: Graph):
    if g.get_class_value(s) == g.get_class_value(t):
        if g.get_class_value(s) == const.MINORITY_VALUE:
            return COLOR_MINORITY
        else:
            return COLOR_MAJORITY
    return COLOR_MIXED


def _get_class_label_color(class_label: str) -> str:
    return COLOR_MINORITY if class_label == const.MINORITY_LABEL \
        else COLOR_MAJORITY if class_label == const.MAJORITY_LABEL \
        else COLOR_MIXED


def _save_plot(fig, fn=None, **kwargs):
    dpi = kwargs.get('dpi', DPI)
    fig.tight_layout()
    if fn is not None and fig is not None:
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        logging.info("%s saved" % fn)
    plt.show()
    plt.close()


def _get_grid_info(total_subplots: int):
    nc = min(MAX_PLOTS_PER_ROW, total_subplots)
    nr = int(np.ceil(nc / MAX_PLOTS_PER_ROW))
    return nc, nr


def _add_class_legend(fig, **kwargs):
    maj_patch = mpatches.Patch(color=COLOR_MAJORITY, label='majority')
    min_patch = mpatches.Patch(color=COLOR_MINORITY, label='minority')
    bbox = kwargs.get('bbox', (1.04, 1))
    loc = kwargs.get('loc', "upper left")
    fig.legend(handles=[maj_patch, min_patch], bbox_to_anchor=bbox, loc=loc)


def plot_graphs(iter_graph: Set[Graph], share_pos=False, fn=None, **kwargs):
    nc, nr = _get_grid_info(len(iter_graph))
    cell_size = kwargs.get('cell_size', DEFAULT_CELL_SIZE)

    fig, axes = plt.subplots(nr, nc, figsize=(nc * cell_size, nr * cell_size), sharex=False, sharey=False)
    node_size = kwargs.get('node_size', 1)
    node_shape = kwargs.get('node_shape', 'o')
    edge_width = kwargs.get('edge_width', 0.02)
    edge_style = kwargs.get('edge_style', 'solid')
    edge_arrows = kwargs.get('edge_arrows', True)
    arrow_style = kwargs.get('arrow_style', '-|>')
    arrow_size = kwargs.get('arrow_size', 2)

    pos = None
    same_n = len(set([g.number_of_nodes() for g in iter_graph])) == 1
    for c, g in enumerate(iter_graph):
        ax = axes[c]
        pos = nx.spring_layout(g) if pos is None or not share_pos or not same_n else pos

        # nodes
        maj = g.graph['class_values'][g.graph['class_labels'].index("M")]
        nodes, node_colors = zip(
            *[(node, COLOR_MAJORITY if data[g.graph['class_attribute']] == maj else COLOR_MINORITY)
              for node, data in g.nodes(data=True)])
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_size=node_size, node_color=node_colors,
                               node_shape=node_shape, ax=ax)

        # edges
        edges = g.edges()
        edges, edge_colors = zip(*[((s, t), _get_edge_color(s, t, g)) for s, t in edges])
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=edges, edge_color=edge_colors,
                               width=edge_width, style=edge_style, arrows=edge_arrows, arrowstyle=arrow_style,
                               arrowsize=arrow_size)

        # final touch
        ax.set_axis_off()
        ax.set_title(g.graph['name'])

    # legend
    _add_class_legend(fig, **kwargs)
    _save_plot(fig, fn, **kwargs)


def plot_distribution(iter_data: Set[pd.DataFrame], x: str, get_x_y_from_df_fnc: callable, fn=None, **kwargs):
    nc, nr = _get_grid_info(len(iter_data))
    cell_size = kwargs.pop('cell_size', DEFAULT_CELL_SIZE)

    hue = kwargs.pop('hue', None)
    sharex = kwargs.pop('sharex', False)
    sharey = kwargs.pop('sharey', False)
    log_scale = kwargs.pop('log_scale', (False, False))
    common_norm = kwargs.pop('common_norm', False)
    ylabel = get_x_y_from_df_fnc.__name__.replace("get_",'').upper()
    ylabel = kwargs.pop('ylabel', ylabel)

    fig, axes = plt.subplots(nr, nc, figsize=(nc * cell_size, nr * cell_size), sharex=sharex, sharey=sharey)

    for cell, df in enumerate(iter_data):
        row = cell // nc
        col = cell % nc

        ax = axes if nr == nc == 1 else axes[cell] if nr == 1 else axes[row, col]

        class_label: str
        for class_label, data in df.groupby(hue):
            total = df[x].sum() if common_norm else data[x].sum()
            xs, ys = get_x_y_from_df_fnc(data, x, total)
            ax.plot(xs, ys, label=class_label, color=_get_class_label_color(class_label), **kwargs)

        if log_scale[0]:
            ax.set_xscale('log')
        if log_scale[1]:
            ax.set_yscale('log')

        if (sharey and col == 0) or (not sharey):
            ax.set_ylabel(ylabel)

        ax.set_xlabel(x)
        ax.set_title(df.name)

    # legend
    _add_class_legend(fig, **kwargs)
    _save_plot(fig, fn, **kwargs)

def plot_powerlaw_fit(iter_data: Set[pd.DataFrame], x: str, kind: str, fn=None, **kwargs):
    nc, nr = _get_grid_info(len(iter_data))
    cell_size = kwargs.pop('cell_size', DEFAULT_CELL_SIZE)

    hue = kwargs.pop('hue', None)
    sharex = kwargs.pop('sharex', False)
    sharey = kwargs.pop('sharey', False)
    log_scale = kwargs.pop('log_scale', (False, False))
    ylabel = kwargs.pop('ylabel', "p(X≥x)" if kind == "ccdf" else "p(X<x)" if kind == 'cdf' else "p(X=x)")
    verbose = kwargs.pop('verbose', False)
    bbox = kwargs.pop('bbox', (1.0, 0.9))

    fig, axes = plt.subplots(nr, nc, figsize=(nc * cell_size, nr * cell_size), sharex=sharex, sharey=sharey)

    for cell, df in enumerate(iter_data):
        row = cell // nc
        col = cell % nc

        ax = axes if nr == nc == 1 else axes[cell] if nr == 1 else axes[row, col]

        class_label: str
        for class_label, data in df.groupby(hue):
            data = data.query(f"{x}>0")
            discrete = data[x].dtype == np.int64
            fit = fit_power_law(data.loc[:,x].values, discrete=discrete, verbose=verbose)

            color = _get_class_label_color(class_label)

            efnc = fit.plot_ccdf if kind == "ccdf" else fit.plot_cdf if kind == 'cdf' else fit.plot_pdf
            fnc = fit.power_law.plot_ccdf if kind == "ccdf" else fit.power_law.plot_cdf if kind == 'cdf' \
                else fit.power_law.plot_pdf

            ax = efnc(label=r"Empirical", ax=ax, color=color, **kwargs)
            ax = fnc(label=f'Powerlaw $\gamma={fit.alpha:.2f}$', linestyle='--', ax=ax, color=color, **kwargs)

            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(handles, labels, loc=3)
            leg.draw_frame(False)

        if log_scale[0]:
            ax.set_xscale('log')
        if log_scale[1]:
            ax.set_yscale('log')

        if (sharey and col == 0) or (not sharey):

            ax.set_ylabel(ylabel)

        ax.set_xlabel(x)
        ax.set_title(df.name)

    # legend
    kwargs['bbox'] = bbox
    _add_class_legend(fig, **kwargs)
    _save_plot(fig, fn, **kwargs)
