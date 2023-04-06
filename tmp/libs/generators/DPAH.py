################################################################
# Systems' dependencies
################################################################
import time
import powerlaw
import numpy as np
import networkx as nx
from collections import Counter
from collections import defaultdict

################################################################
# Constants
################################################################
CLASS = 'm'
LABELS = [0,1] # 0 majority, 1 minority
GROUPS = ['M', 'm']
EPSILON = 0.00001
MODEL_NAME = 'DPAH'

################################################################
# Functions
################################################################

def create(N, fm, d, plo_M, plo_m, h_MM, h_mm, verbose=False, seed=None):
    '''
    Generates a Directed Barabasi-Albert Homophilic network.
    - param N: number of nodes
    - param fm: fraction of minorities
    - param plo_M: power-law outdegree distribution majority class
    - param plo_m: power-law outdegree distribution minority class
    - h_MM: homophily among majorities
    - h_mm: homophily among minorities
    - verbose: if True prints every steps in detail.
    - seed: randommness seed for reproducibility
    '''
    np.random.seed(seed)
    start_time = time.time()
    
    # 1. Init nodes
    nodes, labels, NM, Nm = _init_nodes(N,fm)

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.DiGraph()
    G.graph = {'name':MODEL_NAME, 'class':CLASS, 'groups': GROUPS, 'labels':LABELS}
    nodes, labels, majority, minority = _init_nodes(N, fm)
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G, labels, CLASS)
    
    # 3. Init edges and indegrees
    E = int(round(d * N * (N-1)))
    indegrees = np.zeros(N)
    outdegrees = np.zeros(N)
    
    # 4. Init Activity (out-degree)
    act_M = powerlaw.Power_Law(parameters=[plo_M], discrete=True).generate_random(NM)
    act_m = powerlaw.Power_Law(parameters=[plo_m], discrete=True).generate_random(Nm)
    activity = np.append(act_M, act_m)
    if np.inf in activity:
      activity[activity == np.inf] = 0.0
      activity += 1
    activity /= activity.sum()
    
    # 5. Init homophily
    h_mm = EPSILON if h_mm == 0 else 1-EPSILON if h_mm == 1 else h_mm
    h_MM = EPSILON if h_MM == 0 else 1-EPSILON if h_MM == 1 else h_MM
    homophily = np.array([[h_MM, 1-h_MM],
                          [1-h_mm, h_mm]])
    
    # INIT SUMMARY
    if verbose:
        print("Directed Graph:")
        print("N={} (M={}, m={})".format(N, NM, Nm))
        print("E={} (d={})".format(E, d))
        print("Activity Power-Law outdegree: M={}, m={}".format(plo_M, plo_m))
        print("Homophily: h_MM={}, h_mm={}".format(h_MM, h_mm))
        print(homophily)
        print('')
        
    # 5. Generative process
    tries = 0
    edge_list = defaultdict(list)
    while G.number_of_edges() < E:
      tries += 1
      for source in _pick_sources(N, E, activity):
        # source = _pick_source(N, activity)
        ns = nodes[source]
        target = _pick_target(source, N, labels, indegrees, outdegrees, edge_list, homophily)

        if target is None:
            continue

        nt = nodes[target]

        if not G.has_edge(ns, nt):
            G.add_edge(ns, nt)
            indegrees[target] += 1
            outdegrees[source] += 1
            edge_list[source].append(target)
      
        if G.number_of_edges() >= E:
          break
          
        if verbose:
            ls = labels[source]
            lt = labels[target]
            print("{}->{} ({}{}): {}".format(ns, nt, 'm' if ls else 'M', 'm' if lt else 'M', G.number_of_edges()))

      # outside the for
      if tries > 100 and G.number_of_edges()<E: 
          # it does not find any more new connections
          print("\nEdge density ({}) might differ from {}. N{} fm{} seed{} hMM{} hmm{}\n".format(round(nx.density(G),5), round(d,5),N,fm,seed,h_MM,h_mm))
          break
            
    duration = time.time() - start_time
    if verbose:
        print()
        print(G.graph)
        print(nx.info(G))
        degrees = [d for n,d in G.out_degree()]
        print("min degree={}, max degree={}".format(min(degrees), max(degrees)))
        print(Counter(degrees))
        print(Counter([data[1][CLASS] for data in G.nodes(data=True)]))
        print()
        for k in [0,1]:
            fit = powerlaw.Fit(data=[d for n,d in G.out_degree() if G.node[n][CLASS]==k], discrete=True)
            print("{}: alpha={}, sigma={}, min={}, max={}".format('m' if k else 'M',
                                                                  fit.power_law.alpha, 
                                                                  fit.power_law.sigma, 
                                                                  fit.power_law.xmin, 
                                                                  fit.power_law.xmax))
        print()
        print("--- %s seconds ---" % (duration))

    return G

def _init_nodes(N, fm):
  '''
  Generates random nodes, and assigns them a binary label.
  param N: number of nodes
  param fm: fraction of minorities
  '''
  nodes = np.arange(N)
  majority = int(round(N*(1-fm)))
  minority = N-majority
  minorities = np.random.choice(nodes,minority,replace=False)
  labels = {n:int(n in minorities) for n in nodes}
  return nodes, labels, majority, minority

def _pick_source(N,activity):
    '''
    Picks 1 (index) node as source (edge from) based on activity score.
    '''
    return np.random.choice(a=np.arange(N),size=1,replace=False,p=activity)[0]
  
def _pick_sources(N,E,activity):
    '''
    Picks 1 (index) node as source (edge from) based on activity score.
    '''
    return np.random.choice(a=np.arange(N),size=E,replace=True,p=activity)
    
def _pick_target(source, N, labels, indegrees, outdegrees, edge_list, homophily):
    '''
    Given a (index) source node, it returns 1 (index) target node based on homophily and pref. attachment (indegree).
    The target node must have out_degree > 0 (the older the node in the network, the more likely to get more links)
    '''
    one_percent = N * 1/100.
    if np.count_nonzero(outdegrees)>one_percent:
      targets = [n for n in np.arange(N) if n!=source and n not in edge_list[source]]
    else:
      targets = np.arange(N)
      
    if len(targets) == 0:
        return None
    
    probs = np.array([ homophily[labels[source],labels[n]] * (indegrees[n]+1) for n in targets])
    probs /= probs.sum()
    
    return np.random.choice(a=targets,size=1,replace=True,p=probs)[0]

  
def get_empirical_homophily(fm, e_MM, e_Mm, e_mM, e_mm, pli_M, pli_m, plo_M, plo_m):
  from sympy import symbols
  from sympy import Eq
  from sympy import solve

  # preliminars
  fM = 1-fm
  
  p_MM = e_MM / (e_MM + e_Mm)
  p_mm = e_mm / (e_mm + e_mM)
  
  bo_M = -1/(plo_M + 1)
  bo_m = -1/(plo_m + 1)
  
  bi_M = -1/(pli_M + 1)
  bi_m = -1/(pli_m + 1)

  # equations
  hmm, hmM, hMM, hMm  = symbols('hmm hmM hMM hMm')
  eq1 = Eq( (  (fm * hmm * (1-bi_M)) / ((fm * hmm * (1-bi_M)) + (fM * hmM * (1-bi_m))) )  , p_mm)
  eq2 = Eq( hmm + hmM , 1)
  eq3 = Eq( (  (fM * hMM * (1-bi_m)) / ((fM * hMM * (1-bi_m)) + (fm * hMm * (1-bi_M))) )  , p_MM)
  eq4 = Eq( hMM + hMm , 1)

  solution = solve((eq1,eq2,eq3,eq4), (hmm,hmM,hMM,hMm))
  h_MM, h_mm = solution[hMM], solution[hmm]
  return h_MM, h_mm



#   from sympy import symbols
#   from sympy import Eq
#   from sympy import solve

#   # preliminars
#   fM = 1-fm

#   p_MM = e_MM / (e_MM + e_Mm)
#   p_mm = e_mm / (e_mm + e_mM)
  
#   bo_M = -1/(plo_M + 1)
#   bo_m = -1/(plo_m + 1)

#   bi_M = -1/(pli_M + 1)
#   bi_m = -1/(pli_m + 1)
  
#   # equations
#   hmm, hmM, hMM, hMm  = symbols('hmm hmM hMM hMm')
#   eq1 = Eq( (  (fm  * hmm * (bo_m) * (1-bi_m)) / ((fm  * hmm * (bo_m) * (1-bi_m)) + (fM * hmM * (bo_m) * (1-bi_M))) )  , p_mm)
#   eq2 = Eq( hmm + hmM , 1)
#   eq3 = Eq( (  (fM  * hMM * (bo_M) * (1-bi_M)) / ((fM  * hMM * (bo_M) * (1-bi_M)) + (fm * hMm * (bo_M) * (1-bi_m))) )  , p_MM)
#   eq4 = Eq( hMM + hMm , 1)

#   solution = solve((eq1,eq2,eq3,eq4), (hmm,hmM,hMM,hMm))
#   h_MM, h_mm = solution[hMM], solution[hmm]
#   return h_MM, h_mm