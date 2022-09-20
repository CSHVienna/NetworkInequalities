################################################################
# Dependencies
################################################################
import os
import seaborn as sns
import matplotlib.pyplot as plt

from libs.handlers import utils
from libs.handlers import validations as val

################################################################
# Constants
################################################################
DPI = 300
EXT = 'pdf'

################################################################
# Functions
################################################################

def plot_distributions_across_models(data, kind='kde', d=0.1, fm=0.1, h_MM=0.0, h_mm=0.0, tc=0.1, plo_M=0.2, plo_m=0.2, output=None):
  val.validate_displot_kind(kind)
  val.validate_not_none(d=d, fm=fm, h_MM=h_MM, h_mm=h_mm, tc=tc, plo_M=plo_M, plo_m=plo_m)
  MODEL_ACTIVITY_DENSITY = ['DPA','DH','DPAH']
  data = data.query("fm==@fm and ((h_MM==@h_MM and h_mm==@h_mm and name!='DPA') or name=='DPA') and ((tc==@tc and name=='PATCH') or name!='PATCH') and ((plo_M==@plo_M and plo_m==@plo_m and name in @MODEL_ACTIVITY_DENSITY) or name not in @MODEL_ACTIVITY_DENSITY) and ((d==@d and name in @MODEL_ACTIVITY_DENSITY) or name not in @MODEL_ACTIVITY_DENSITY)").copy()
  
  # plot
  fg = sns.displot(
      data=data, x="value", hue="label", col="name", row="metric",
      kind=kind, height=2, aspect=1.,
      facet_kws=dict(margin_titles=True, sharex=False, sharey=kind=='ecdf'),
  )
  
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
  
  
# def plot_degree_distribution_multiple_graphs(df_distributions, kind='kde', subset_gen=None, subset_hs=None, subset_fm=None, output=None):
#   # kind: kde, ecdf, hist
  
#   # only a subste of homophily values
#   if subset_gen is not None:
#     q = " or ".join([f"name=='{name}'"for name in subset_gen])
#     data = df_distributions.query(q).copy()
#   else:
#     data = df_distributions.copy()
    
#   # only a subste of homophily values
#   if subset_hs is not None:
#     q = " or ".join([f"(h_MM=={h_MM} and h_mm=={h_mm})"for (h_MM,h_mm) in subset_hs])
#     data = data.query(q).copy()
#   else:
#     data = data.copy()
    
#   # only a subste of fm values
#   if subset_fm is not None:
#     q = " or ".join([f"fm=={fm}"for fm in subset_fm])
#     data = data.query(q).copy()
#   else:
#     data = data.copy()
    
#   # homophily column
#   data.loc[:,'h'] = data.apply(lambda row:f"hMM:{row.h_MM} | hmm:{row.h_mm}", axis=1)
  
#   # plot
#   fg = sns.displot(
#       data=data, x="degree", hue="label", col="h", row="fm",
#       kind=kind, height=2, aspect=1.,
#       facet_kws={"margin_titles": True},
#   )
  
#   [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
#   fg.set_titles(col_template = '{col_name}') #row_template = '{row_name}', 
#   plt.subplots_adjust(wspace=0.1, hspace=0.1)
  
#   plt.show()
#   plt.close()