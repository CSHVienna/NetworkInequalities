################################################################
# Dependencies
################################################################
import os
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

from libs.handlers import utils
from libs.handlers import io
from libs.handlers import validations as val

################################################################
# Constants
################################################################
DPI = 300
EXT = 'pdf'
INACTIVE_COLOR = 'lightgrey'

################################################################
# Functions
################################################################
  
def plot_distributions_across_models(data, kind='kde', output=None, **kws):
  val.validate_displot_kind(kind)
  df = utils.dataframe_sample(data, **kws)
  fm,m,d,h_MM,h_mm,tc,plo_M,plo_m = utils.get_main_params(**kws)
  
  def _check_empty(data, **kws):
    ax = plt.gca()
    if data['value'].isnull().values.any():
      set_inactive_ax(ax)
      return
    
  # plot
  fg = sns.displot(
      data=df, x="value", hue="label", col="name", row="metric",
      kind=kind, height=2, aspect=1.,
      facet_kws=dict(margin_titles=True, sharex=False, sharey=kind=='ecdf'),
  )
  fg.map_dataframe(_check_empty)
  
  [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
  fg.set_titles(col_template = '{col_name}', row_template = '{row_name}') 
  title = f"{kind.upper()} (d{d} | fm{fm} | hMM{h_MM} | hmm{h_mm} | tc{tc} | ploM{plo_M} | plom{plo_m})"
  title = title.replace("fm","f$_m$").replace("hMM","h$_{MM}$").replace("hmm","h$_{mm}$").replace("ploM","plo$_{M}$").replace("plom","plo$_{m}$")
  plt.suptitle(title, y=1.03)
  
  if output is not None:
    fn = os.path.join(output, f'distribution_across_generators_{kind}_d{d}_fm{fm}_hMM{h_MM}_hmm{h_mm}_tc{tc}_ploM{plo_M}_plom{plo_m}.{EXT}')
    fg.savefig(fn, dpi=DPI, bbox_inches='tight')
    utils.info(f"{fn} saved!")
    
  plt.show()
  plt.close()
  
def set_inactive_ax(ax):
  ax.tick_params(axis='x', colors=INACTIVE_COLOR)
  # ax.tick_params(axis='y', colors=INACTIVE_COLOR)
  ax.spines['top'].set_color(INACTIVE_COLOR)
  ax.spines['left'].set_color(INACTIVE_COLOR)
  ax.spines['right'].set_color(INACTIVE_COLOR)
  ax.spines['bottom'].set_color(INACTIVE_COLOR)
  
def plot_inequality_across_models(data, output=None, **kws):
  df = utils.dataframe_sample(data, **kws)
  fm,m,d,h_MM,h_mm,tc,plo_M,plo_m = utils.get_main_params(**kws)
  
  def _inequality(data, **kws):
    ax = plt.gca()
    if data['value'].isnull().values.any():
      set_inactive_ax(ax)
      return
    x = utils.get_rank_range()
    y = [utils.gini(data.query("rank<=@i").value.values) for i in x]
    ax.plot(x,y)
                 
  # plot
  fg = sns.FacetGrid(
      data=df, col="name", row="metric",
      height=2, aspect=1.,
      margin_titles=True,
  )
  fg.map_dataframe(_inequality, x='rank')
  
  [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
  fg.set_titles(col_template = '{col_name}', row_template = '{row_name}') 
  title = f"Rank Inequality (d{d} | fm{fm} | hMM{h_MM} | hmm{h_mm} | tc{tc} | ploM{plo_M} | plom{plo_m})"
  title = title.replace("fm","f$_m$").replace("hMM","h$_{MM}$").replace("hmm","h$_{mm}$").replace("ploM","plo$_{M}$").replace("plom","plo$_{m}$")
  plt.suptitle(title, y=1.03)
  fg.set_ylabels("Gini")
  fg.set_xlabels("Top-k% rank")
  
  if output is not None:
    for ext in [EXT,'png']:
      fn = os.path.join(output, f'inequality_across_generators_d{d}_fm{fm}_hMM{h_MM}_hmm{h_mm}_tc{tc}_ploM{plo_M}_plom{plo_m}.{ext}')
      fg.savefig(fn, dpi=DPI, bbox_inches='tight')
      utils.info(f"{fn} saved!")
    
  plt.show()
  plt.close()
  
def plot_inequity_across_models(data, output=None, **kws):
  df = utils.dataframe_sample(data, **kws)
  fm,m,d,h_MM,h_mm,tc,plo_M,plo_m = utils.get_main_params(**kws)
  
  def _inequity(data, **kws):
    ax = plt.gca()
    if data['value'].isnull().values.any():
      set_inactive_ax(ax)
      return
    colors = mcp.gen_color(cmap="tab10",n=10)
    x = utils.get_rank_range()
    y = [utils.percent_minorities(data.query("rank<=@i")) for i in x]
    ax.plot(x,y,c=colors[1])
    me, interpretation = utils.get_inequity_mean_error(y,fm)
    ax.text(s=f"ME={me:.2f}\n({interpretation})", x=0.5, y=0.5, va='center', ha='center', transform=ax.transAxes)
                 
  # plot
  fg = sns.FacetGrid(
      data=df, col="name", row="metric",
      height=2, aspect=1.,
      margin_titles=True,
  )
  fg.map_dataframe(_inequity, x='rank')
  fg.refline(y=fm, lw=1)
  
  # [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
  fg.set_titles(col_template = '{col_name}', row_template = '{row_name}') 
  title = f"Rank Inequity (d{d} | fm{fm} | hMM{h_MM} | hmm{h_mm} | tc{tc} | ploM{plo_M} | plom{plo_m})"
  title = title.replace("fm","f$_m$").replace("hMM","h$_{MM}$").replace("hmm","h$_{mm}$").replace("ploM","plo$_{M}$").replace("plom","plo$_{m}$")
  plt.suptitle(title, y=0.99)
  fg.set_ylabels("Minority size (%)")
  fg.set_xlabels("Top-k% rank")
  plt.tight_layout()
  
  if output is not None:
    for ext in [EXT, 'png']:
      fn = os.path.join(output, f'inequity_across_generators_d{d}_fm{fm}_hMM{h_MM}_hmm{h_mm}_tc{tc}_ploM{plo_M}_plom{plo_m}.{ext}')
      fg.savefig(fn, dpi=DPI, bbox_inches='tight')
      utils.info(f"{fn} saved!")
    
  plt.show()
  plt.close()
  
  
def plot_network_across_models(data, output=None, **kws):
  import glob
  import os
  
  fm,m,d,h_MM,h_mm,tc,plo_M,plo_m = utils.get_main_params(**kws)
  files = set()
  
  # PAH
  files |= set(glob.glob(os.path.join(data,'*',f'*_m{m}_fm{fm}_hMM{h_MM}_hmm{h_mm}_seed*.gpickle')))
  
  # PATCH
  files |= set(glob.glob(os.path.join(data,'*',f'*_m{m}_fm{fm}_hMM{h_MM}_hmm{h_mm}_tc{tc}_seed*.gpickle')))
  
  # DPAH
  files |= set(glob.glob(os.path.join(data,'*',f'*fm{fm}_d{d}_hMM{h_MM}_hmm{h_mm}_ploM{plo_M}_plom{plo_m}_seed*.gpickle')))
  
  # DPA
  files |= set(glob.glob(os.path.join(data,'*',f'*fm{fm}_d{d}_ploM{plo_M}_plom{plo_m}_seed*.gpickle')))
  
  # DH
  files |= set(glob.glob(os.path.join(data,'*',f'*fm{fm}_d{d}_hMM{h_MM}_hmm{h_mm}_ploM{plo_M}_plom{plo_m}_seed*.gpickle')))
  
  def _get_edge_color(colors, s, t, graph_metadata):
    if G.nodes[s][G.graph['label']]==0 and G.nodes[s][G.graph['label']]==G.nodes[t][G.graph['label']]:
      return colors[0]
    if G.nodes[s][G.graph['label']]==1 and G.nodes[s][G.graph['label']]==G.nodes[t][G.graph['label']]:
      return colors[1]
    return 'grey'
  
  files = sorted(files)
  
  nc = 5
  nr = 1
  size = 3
  fig, axes = plt.subplots(nr, nc, figsize=(nc*size, nr*size), sharex=False, sharey=False)
  
  colors = mcp.gen_color(cmap="tab10",n=10)
  for c, fn in enumerate(files):
    G = io.read_gpickle(fn)
    ax = axes[c]
    pos = nx.spring_layout(G)
    
    # nodes
    nodes_data = G.nodes(data=True)
    nodes = [n for n,data in nodes_data]
    node_colors = [colors[data[G.graph['label']]] for n,data in nodes_data]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=1, node_color=node_colors, node_shape='o', ax=ax)

    # edges
    edges = G.edges()
    edge_colors = [_get_edge_color(colors, s, t, G.graph) for s,t in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=0.02, edge_color=edge_colors, style='solid',
                           arrows=True, arrowstyle='-|>', arrowsize=2)
    
    # final touch
    ax.set_axis_off()
    ax.set_title(G.graph['name'])
   
  if output is not None:
    for ext in [EXT,'png']:
      fn = os.path.join(output, f'networks_across_generators_d{d}_fm{fm}_hMM{h_MM}_hmm{h_mm}_tc{tc}_ploM{plo_M}_plom{plo_m}.{ext}')
      fig.savefig(fn, dpi=DPI, bbox_inches='tight')
      utils.info(f"{fn} saved!")
    
  plt.show()
  plt.close()