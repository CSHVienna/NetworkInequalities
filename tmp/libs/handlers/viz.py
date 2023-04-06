################################################################
# Dependencies
################################################################
import os
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from tmp.libs.handlers import io, utils
from tmp.libs.handlers import rank
from tmp.libs.handlers import network as nw
from tmp.libs.handlers import validations as val
from tmp.libs.generators import model

################################################################
# Constants
################################################################
DPI = 300
EXT = 'pdf'
INACTIVE_COLOR = 'lightgrey'
LS_EMPIRICAL = '-'
LS_FIT = '--'
COLOR_EMPIRICAL = 'tab:blue'
COLOR_FIT = 'tab:orange'
COLOR_MAJORITY = 'tab:blue'
COLOR_MINORITY = 'tab:red'

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
    for ext in [EXT,'png']:
      fn = os.path.join(output, f'distribution_across_generators_{kind}_d{d}_fm{fm}_hMM{h_MM}_hmm{h_mm}_tc{tc}_ploM{plo_M}_plom{plo_m}.{ext}')
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
    y = [rank.gini(data.query("rank<=@i").value.values) for i in x]
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
  
def _plot_inequity(data, **kws):
  ax = plt.gca() if 'ax' not in kws else kws.pop('ax')
  
  if data['value'].isnull().values.any() or data.shape[0]==0:
    set_inactive_ax(ax)
    return
  
  x = utils.get_rank_range()
  y = [rank.percent_minorities(data.query("rank<=@i")) for i in x]
  color = kws.pop('color') if 'color' in kws else COLOR_EMPIRICAL
  ax.plot(x,y,c=color, **kws)
  
  if 'label' in kws:
    if kws['label']=='Empirical':
      # empirical vs fit plot 
      fm = data.fm.unique()[0]
      ax.axhline(fm,ls='--',c='grey',lw=1.0)
  else:
    # general plot
    fm = data.fm.unique()[0]
    me, interpretation = rank.get_inequity_mean_error(y, fm)
    ax.text(s=f"ME={me:.2f}\n({interpretation})", x=0.5, y=0.5, va='center', ha='center', transform=ax.transAxes)
    
def plot_inequity_across_models(data, output=None, **kws):
  df = utils.dataframe_sample(data, **kws)
  fm,m,d,h_MM,h_mm,tc,plo_M,plo_m = utils.get_main_params(**kws)
                 
  # plot
  fg = sns.FacetGrid(
      data=df, col="name", row="metric",
      height=2, aspect=1.,
      margin_titles=True,
  )
  fg.map_dataframe(_plot_inequity)
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
  
  def _get_edge_color(s, t, graph_metadata):
    if G.nodes[s][G.graph['class']]==0 and G.nodes[s][G.graph['class']]==G.nodes[t][G.graph['class']]:
      return COLOR_MAJORITY
    if G.nodes[s][G.graph['class']]==1 and G.nodes[s][G.graph['class']]==G.nodes[t][G.graph['class']]:
      return COLOR_MINORITY
    return INACTIVE_COLOR
  
  files = sorted(files)
  
  nc = 5
  nr = 1
  size = 3
  fig, axes = plt.subplots(nr, nc, figsize=(nc*size, nr*size), sharex=False, sharey=False)
  
  # colors = mcp.gen_color(cmap="tab10",n=10)
  for c, fn in enumerate(files):
    G = io.read_gpickle(fn)
    ax = axes[c]
    pos = nx.spring_layout(G)
    
    # nodes
    nodes_data = G.nodes(data=True)
    nodes = [n for n,data in nodes_data]
    node_colors = [COLOR_MAJORITY if data[G.graph['class']]==G.graph['labels'][0] else COLOR_MINORITY for n,data in nodes_data]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=1, node_color=node_colors, node_shape='o', ax=ax)

    # edges
    edges = G.edges()
    edge_colors = [_get_edge_color(s, t, G.graph) for s,t in edges]
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
  
  
def plot_empirical_vs_fit(empirical_path, fit_path, kind='distributions', metric='pagerank', output=None, verbose=False):
  import glob
  import os
  from tmp.libs.generators.model import MODEL_NAMES
  
  NNODES = 1000
  MODEL_NAMES = MODEL_NAMES
  val.validate_empirical_vs_fit_kind(kind)
  
  empirical_files = glob.glob(os.path.join(empirical_path,'*.gpickle'))
  nr = len(empirical_files)
  nc = len(MODEL_NAMES)
  size = 3
  
  fig, axes = plt.subplots(nr, nc, figsize=(nc*size, nr*size), sharex=True, sharey=True)
  
  for r,fn_e in enumerate(sorted(empirical_files)):
    # load empirical GRAPH
    Ge = io.read_gpickle(fn_e)
    is_directed = Ge.is_directed()
    e_MM, e_Mm, e_mM, e_mm = nw.get_edge_counts(Ge)
    Ee = Ge.number_of_edges()
      
    # load node distributions
    fn_ed = fn_e.replace('.gpickle','.csv')
    if io.exists(fn_ed):
      dfe = io.read_csv(fn_ed)
    else:
      dfe = utils.get_node_distributions_as_dataframe(Ge)
      io.to_csv(dfe, fn_ed)
   
    if dfe['network_id'].isnull().values.any():
      dfe.loc[:,'network_id'] = 1
      
    # load fits (or do fitting)
    for c,generator_name in enumerate(MODEL_NAMES):
      
      ax = axes[r,c]
      ax.set_title(generator_name if r==0 else '')
      ax.set_ylabel('y' if c==0 else '')

      if c==nc-1:
        ax2 = ax.twinx()
        ax2.set_ylabel(f"{'d' if is_directed else ''}{Ge.graph['name']}", rotation=270, labelpad=8)
        ax2.set_yticks([])
        
      if is_directed and generator_name in ['PAH','PATCH'] or not is_directed and generator_name in ['DPA','DH','DPAH']:
        set_inactive_ax(ax)
        if c==nc-1:
          set_inactive_ax(ax2)
        continue
        
      for k in ['fm','d','pl_M','pl_m','pli_M','pli_m','plo_M','plo_m']:
        if k in Ge.graph and Ge.graph[k] is not None:
          precision = 2
          while True:
            if round(Ge.graph[k],precision) > 0:
              break
            precision+=1
          Ge.graph[k] = round(Ge.graph[k],precision)
       
      Ne = Ge.graph['N']
      Ge.graph['N'] = NNODES
      fn = model.get_filename(Ge, 'gpickle').replace(Ge.graph['name'], f"{Ge.graph['name']}_{generator_name}")
      fn_f = os.path.join(fit_path, Ge.graph['name'], fn)
      val.validate_path(fn_f)
      fn_fd = fn_f.replace('gpickle','csv')
      Ge.graph['N'] = Ne
      
      try:
        if io.exists(fn_f) and io.exists(fn_fd):
          Gf = io.read_gpickle(fn_f)
          dff = io.read_csv(fn_fd)
        else:
          obj = nw.get_hyperparams(Ge, generator_name, verbose=verbose)
          obj['N']=max((min((NNODES,Ge.graph['N'])),100))
          Gf = model.create(obj, seed=utils.get_random_seed())
          dff = utils.get_node_distributions_as_dataframe(Gf)
          dff.loc[:,'network_id'] = 1
          io.to_gpickle(Gf, fn_f)
          io.to_csv(dff, fn_fd)
      except Exception as ex:
        utils.error(f"plot_empirical_vs_fit | viz.py | {ex}")
        
      fe_MM, fe_Mm, fe_mM, fe_mm = nw.get_edge_counts(Gf)
      Ef = Gf.number_of_edges()
      e_MM/=Ee
      e_Mm/=Ee
      e_mM/=Ee
      e_mm/=Ee
      fe_MM/=Ef
      fe_Mm/=Ef
      fe_mM/=Ef
      fe_mm/=Ef
      error_fit = abs(e_MM-fe_MM)+abs(e_Mm-fe_Mm)+abs(e_mM-fe_mM)+abs(e_mm-fe_mm)
      ax.text(s=f'PAE={error_fit:.2f}', x=0.5, y=1.0, va='top', ha='center', transform=ax.transAxes)
      
      if generator_name not in ['DPA']:
        hMM,hmm,tc = nw.get_homophily_and_triadic_closure(Ge, generator_name=generator_name, verbose=verbose)
        hMMf,hmmf,tcf = nw.get_homophily_and_triadic_closure(Gf, generator_name=generator_name, verbose=verbose)
        NONE = [None,np.nan]
        hMM = round(hMM,2) if hMM not in NONE else hMM
        hmm = round(hmm,2) if hmm not in NONE else hmm
        tc = round(tc,2) if tc not in NONE else tc
        hMMf = round(hMMf,2) if hMMf not in NONE else hMMf
        hmmf = round(hmmf,2) if hmmf not in NONE else hmmf
        tcf = round(tcf,2) if tcf not in NONE else tcf
        ax.text(s='h$_{MM}$='+str(hMM)+' h$_{mm}$='+str(hmm)+' '+(f"tc={tc}" if tc not in NONE else ''), x=0.5, y=0.90, va='top', ha='center', transform=ax.transAxes, color=COLOR_EMPIRICAL)
        ax.text(s='h$_{MM}$='+str(hMMf)+' h$_{mm}$='+str(hmmf)+' '+(f"tc={tcf}" if tcf not in NONE else ''), x=0.5, y=0.80, va='top', ha='center', transform=ax.transAxes, color=COLOR_FIT)
    
      data_e = rank.apply_rank(utils.flatten_dataframe_by_metric(dfe))
      data_f = rank.apply_rank(utils.flatten_dataframe_by_metric(dff))
      if kind=='inequity':
        _plot_inequity(data_e.query('metric==@metric'), ax=ax, color=COLOR_EMPIRICAL, label='Empirical')
        _plot_inequity(data_f.query('metric==@metric'), ax=ax, color=COLOR_FIT, label='Fit')
      
  plt.suptitle(f"Empirical vs. Fit | {kind}", y=0.9)
  
  if output is not None:
    for ext in [EXT,'png']:
      fn = os.path.join(output, f'model_fitting.{ext}')
      fig.savefig(fn, dpi=DPI, bbox_inches='tight')
      utils.info(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
  
####################################################################################################################################

def plot_degree_distritbution_per_class(data, **kws):
    df_min = data.query("minority==1").copy()
    df_maj = data.query("minority==0").copy()
    ax = plt.gca()
    color = kws.pop('color')
    ecdf = kws.pop('ecdf')
    fnc = sns.ecdfplot if ecdf else sns.kdeplot
    fnc(df_min, ax=ax, x='degree', label='minority', ls='--', color='black')
    fnc(df_maj, ax=ax, x='degree', label='majority', ls='--', color='grey')
    return 

def plot_degree_distributions(G, Gf, ecdf=False, verbose=False):
    minclass = [obj[G.graph['class']]==G.graph['labels'][1] for n,obj in G.nodes(data=True)]
    minclassf = [obj[Gf.graph['class']]==Gf.graph['labels'][1] for n,obj in Gf.nodes(data=True)]
    
    if G.is_directed():    
        df = pd.DataFrame({'kind':'empirical', 'direction':'indegree', 'minority':minclass, 'degree':[d for n,d in G.in_degree()]})
        df = pd.concat([df, pd.DataFrame({'kind':'empirical', 'direction':'outdegree', 'minority':minclass, 'degree':[d for n,d in G.out_degree()]})])
        df = pd.concat([df, pd.DataFrame({'kind':'fit', 'direction':'indegree', 'minority':minclassf, 'degree':[d for n,d in Gf.in_degree()]})])
        df = pd.concat([df, pd.DataFrame({'kind':'fit', 'direction':'outdegree', 'minority':minclassf, 'degree':[d for n,d in Gf.out_degree()]})])
    else:
        df = pd.DataFrame({'kind':'empirical', 'direction':'empirical', 'minority':minclass, 'degree':[d for n,d in G.degree()]})
        df = pd.concat([df, pd.DataFrame({'kind':'fit', 'direction':'fit', 'minority':minclassf, 'degree':[d for n,d in Gf.degree()]})])
        
    fg = sns.FacetGrid(data=df, col='direction', hue='kind', palette='tab10', sharex=False, sharey=False)
    fg.map_dataframe(sns.kdeplot if not ecdf else sns.ecdfplot, x='degree')
    #fg.map_dataframe(plot_degree_distritbution_per_class, ecdf=ecdf)
    
    fg.add_legend()
    plt.suptitle(f"{G.graph['name']} & {Gf.graph['name']}", y=1.01);
    [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
    fg.set_titles(col_template = '{col_name}', row_template = '{row_name}') 
    for ax in fg.axes.flatten():
        ax.set_xscale('log')
    
    hMM,hmm,_ = nw.get_homophily_and_triadic_closure(G, generator_name=G.graph['model'], verbose=verbose)
    hMMf,hmmf,_ = nw.get_homophily_and_triadic_closure(Gf, generator_name=G.graph['model'], verbose=verbose)
    ax = fg.axes.flatten()[0]
    ax.text(s=f"N={G.number_of_nodes()}, E={G.number_of_edges()}, d={nx.density(G):.5f}, hMM={hMM:.2f}, hmm={hmm:.2f}", x=0.0, y=-0.35, va='bottom', ha='left', transform=ax.transAxes, color='tab:blue');
    ax.text(s=f"N={Gf.number_of_nodes()}, E={Gf.number_of_edges()}, d={nx.density(Gf):.5f}, hMM={hMMf:.2f}, hmm={hmmf:.2f}", x=0.0, y=-0.45, va='bottom', ha='left', transform=ax.transAxes, color='tab:orange');
    
def plot_evaluation_metrics_vs_sample_size(df_results, G, Gf):
    color = {'empirical':'tab:blue', 'fit':'tab:orange'}
    metrics = [c for c in df_results.columns if c not in ['kind','sample_size']]
    nr = 1
    nc = len(metrics)
    s = 3
    fig, axes = plt.subplots(1,nc,figsize=(nc*s, nr*s),sharex=True,sharey=True)

    for c, metric in enumerate(metrics):
        ax = axes[c]
        ax.set_title(metric)
        ax.set_ylabel('')
        ax.axhline(0.5, lw=1, c='grey', ls='--')
        for kind, tmp in df_results.groupby('kind'):
            _ = sns.lineplot(data=tmp, x='sample_size', y=metric, color=color[kind], label=kind, ax=ax, legend=metric=='rocauc')

    plt.suptitle(f"{G.graph['name']} & {Gf.graph['name']}", y=1.01);