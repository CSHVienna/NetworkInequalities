"""
Generators for BA homophily network with minority, majority,
and triadic closure

written by: 
- 2018 Fariba Karimi
- 2021 Jan Bachmann
- 2022 Lisette Espin-Noboa
"""

################################################################
# Dependencies
################################################################
from typing import Callable, Union, Set
from collections import defaultdict
import random
import numpy as np
import networkx as nx

################################################################
# Constants
################################################################
MODEL_NAME = 'PATCH'
CLASS = 'm'
LABELS = [0,1] # 0 majority, 1 minority
GROUPS = ['M', 'm']
EPSILON = 0.00001

################################################################
# Main
################################################################
def create(
        N: int,
        m: int,
        fm: float,
        h_MM: float,
        h_mm: float,
        triadic_closure: float,
        seed: float = None,
        _on_edge_added: Union[None, Callable[[nx.Graph, int, int], None]] = None,
        _on_tc_edge_added: Union[None, Callable[[nx.Graph, int, int], None]] = None) -> nx.Graph:
    """Return random undigraph using BA preferential attachment model
    accounting for specified homophily and triadic closure.

    A undigraph of n nodes is grown by attaching new nodes each with m edges.
    Each edge is either drawn by the triadic closure or a homophily induced
    preferential attachment procedure. For the former a candidate is chosen
    uniformly at random from all triad-closing edges (of the new node).
    Otherwise (or if there are no triads to close) edges are preferentially
    attached to existing nodes with high degree.
    These connections are established by linking probability which
    depends on the connectivity of sites and the homophily (similarities).
    For high homophily nodes with equal group membership
    are more likely to connect.
    homophily varies ranges from 0 to 1.

    Parameters
    ----------
    N : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    minority_fraction : float
        fraction of minorities in the network

    homophily: float
        value between 0 to 1. similarity between nodes. if nodes have same attribute
        their homophily (distance) is smaller.

    triadic_closure: float
        Value between 0 to 1. The probability of a new edge to close a triad.
        If met, an edge to current neighbor's neighbors is drawn uniformly at random.
        Otherwise, the edge is added based on the preferential attachment and homophily dynamic.
        Set to 0. to deactivate this dynamic.

    _on_edge_added: Union[None, Callable[[nx.Graph, int, int], None] (default=None)
        Can be used to inject a function that is called after an edge is added.
        The function is expected to take the undigraph and two ints as input.
        The latter describe the source and the target node of the newly created edge.

    _on_tc_edge_added: Union[None, Callable[[nx.Graph, int, int], None] (default=None)
        Can be used to inject a function that is called after a triadic closure edge is added.
        The function is expected to take the undigraph and two ints as input.
        The latter describe the source and the target node of the newly created edge.

    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a undigraph with with m nodes and no edges.

    References
    ----------
    .. [1] A. L. Barabasi and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    np.random.seed(seed)
    m = max(1, m)
    
    # 1. Init nodes
    G = nx.Graph()
    G.graph = {'name':MODEL_NAME, 'class':CLASS, 'groups': GROUPS, 'labels':LABELS}
    nodes, labels, majority, minority = _init_nodes(N, fm)
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G, labels, CLASS)
  
    h_mm = EPSILON if h_mm == 0 else 1-EPSILON if h_mm == 1 else h_mm
    h_MM = EPSILON if h_MM == 0 else 1-EPSILON if h_MM == 1 else h_MM
    homophily = np.array([[h_MM, 1-h_MM],[1-h_mm, h_mm]])
    
    # Node to be added, init to m to start with m pre-existing, unconnected nodes
    source = m

    while source < N: # Iterate until N nodes were added
        targets_tc = defaultdict(int)
        targets_pah = set(range(source))
        tc_edge = False # Remember if edge was caused by TC
        for idx_m in range(m):
            tc_edge = False
            tc_prob = random.random()

            # Choose next target based on TC or PA+H
            if tc_prob < triadic_closure and len(targets_tc) > 0:
                # TODO: Find a better way as the conversion takes O(N)
                # Sample TC target based on its occurrence frequency
                # target = random.choices(
                #     list(targets_tc.keys()), # Nodes themselves
                #     weights=list(targets_tc.values()), # Weight by frequency
                #     k=1)[0] # Select k=1 target
                
                target_list, probs = zip(*[(t,w) for t,w in targets_tc.items()])
                probs = np.array(probs)
                target = np.random.choice(a=target_list,size=1,replace=False,p=probs/probs.sum())[0]
                tc_edge = True
            else:
                target = _pick_pa_h_target(G, source, targets_pah, labels, homophily)
                
            # [Perf.] No need to update target dicts if we are adding the last edge
            if idx_m < m-1:
                # Remove target target candidates of source
                targets_pah.discard(target)
                if target in targets_tc:
                    del targets_tc[target] # Remove target from TC candidates

                # Incr. occurrence counter for friends of new friend
                for neighbor in G.neighbors(target):
                    # G[source] gives direct access (O(1)) to source's neighbors
                    # G.neighbors(source) returns an iterator which would
                    # need to be searched iteratively
                    if neighbor not in G[source]:
                        targets_tc[neighbor] += 1

            # Finally add edge to undigraph
            G.add_edge(source, target)

            # Call event handlers if present
            if _on_edge_added is not None:
                _on_edge_added(G, source, target)
            if _on_tc_edge_added is not None and tc_edge:
                _on_tc_edge_added(G, source, target)

        # Activate node as potential target and select next node
        source += 1

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
  
def _pick_pa_h_target(
        G: nx.Graph,
        source: int,
        target_set: Set[int],
        labels: Set[bool],
        homophily: float) -> Union[int, None]:
    """Picks a random target node based on the homophily/preferential attachment dynamic.

    Args:
        G (nx.Graph): Current undigraph instance
        source (int): Newly added node
        target_set (Set[int]): Potential target nodes in the undigraph
        minority_nodes (Set[bool]): Set of minority nodes
        homophily (float): Effect of homophily.
            If high, nodes with same group membership are more likely to connect to one another.

    Returns:
        int or None: Target node that an edge should be added to
    """
    # Collect probabilities to connect to each node in target_list
    target_list = [t for t in target_set if t!=source and t not in nx.neighbors(G,source)]
    probs = np.array([ homophily[labels[source],labels[target]] * (G.degree(target)+EPSILON) for target in target_list])
    probs /= probs.sum()
    return np.random.choice(a=target_list,size=1,replace=False,p=probs)[0]

