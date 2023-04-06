"""
Generators for BA homophilic network with minority and majority.


written by: Fariba Karimi
Date: 22-01-2016
"""

################################################################
# Dependencies
################################################################
import networkx as nx
from collections import defaultdict
import random
import bisect
import copy
import numpy as np

################################################################
# Constants
################################################################
MODEL_NAME = 'PAH'
CLASS = 'm'
LABELS = [0, 1] # 0 majority, 1 minority
GROUPS = ['M', 'm']
EPSILON = 0.00001

################################################################
# Main
################################################################

def create(N, m, fm, h_MM, h_mm, d=None, seed=None):
    np.random.seed(seed)
    m = max(1, m)
    
    # expected values
    MORE_E = 0
    if d is not None:
        EXPECTED_E = int(round(d*(N*(N-1)/2.)))
        MIN_E = (N-m) * m
        MORE_E = EXPECTED_E - MIN_E

    # Add node attributes
    G = nx.Graph()
    G.graph = {'name':MODEL_NAME, 'class':CLASS, 'groups': GROUPS, 'labels':LABELS}
    nodes, labels, majority, minority = _init_nodes(N, fm)
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G, labels, CLASS)
    
    # homophily
    h_mm = EPSILON if h_mm == 0 else 1-EPSILON if h_mm == 1 else h_mm
    h_MM = EPSILON if h_MM == 0 else 1-EPSILON if h_MM == 1 else h_MM
    homophily = np.array([[h_MM, 1-h_MM],
                          [1-h_mm, h_mm]])
    
    target_set = set(range(m))
    source = m

    while source < N:

        targets = _pick_targets(G, source, target_set, homophily, labels, m)
        
        if targets is not None:          
          if len(targets)>0:  # if the node does  find the neighbor
            G.add_edges_from(zip([source] * m, targets))
            target_set.add(source)
          
        source += 1

    ### Fulfilling density while keeping min E = (N*m)
    if MORE_E > 0:
      m = 1
      counter = 0
      while counter < MORE_E:
          source = random.choice(nodes)
          targets = list(_pick_targets(G, source, target_set, homophily, labels, m))

          if targets is not None:
            if len(targets) > 0:
              G.add_edges_from(zip([source] * m, targets))
              counter += m

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

def _pick_targets(G, source, target_set, homophily, labels, m):
  target_list = [t for t in target_set if t!=source and t not in nx.neighbors(G,source)]
  probs = np.array([ homophily[labels[source],labels[target]] * (G.degree(target)+EPSILON) for target in target_list])
  probs /= probs.sum()
  return np.random.choice(a=target_list,size=m,replace=False,p=probs)

  
def get_empirical_homophily(fm, e_MM, e_Mm, e_mM, e_mm, pl_M, pl_m):
  from sympy import symbols
  from sympy import Eq
  from sympy import solve
  
  # preliminars
  fM = 1-fm
  
  e_mix = e_Mm + e_mM
  
  p_MM = e_MM / (e_MM+int(e_mix/2))
  p_mm = e_mm / (e_mm+int(e_mix/2))
  
  b_M = -1/(pl_M + 1)
  b_m = -1/(pl_m + 1)
  
  # equations
  hmm, hMM, hmM, hMm  = symbols('hmm hMM hmM hMm')
  eq1 = Eq( (fm * 1 * hmm * (1-b_M)) / ((fm * hmm * (1-b_M)) + (fM * hmM * (1-b_m))), p_mm)
  eq2 = Eq( hmm + hmM, 1)
  
  eq3 = Eq( (fM * 1 * hMM * (1-b_m)) / ((fM * hMM * (1-b_m)) + (fm * hMm * (1-b_M))), p_MM)
  eq4 = Eq( hMM + hMm , 1)

  solution = solve((eq1,eq2,eq3,eq4), (hmm,hmM,hMM,hMm))
  h_MM, h_mm = solution[hMM], solution[hmm]
  return h_MM, h_mm