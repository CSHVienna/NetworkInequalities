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
    """Return random graph using BA preferential attachment model
    accounting for specified homophily and triadic closure.

    A graph of n nodes is grown by attaching new nodes each with m edges.
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
        The function is expected to take the graph and two ints as input.
        The latter describe the source and the target node of the newly created edge.

    _on_tc_edge_added: Union[None, Callable[[nx.Graph, int, int], None] (default=None)
        Can be used to inject a function that is called after a triadic closure edge is added.
        The function is expected to take the graph and two ints as input.
        The latter describe the source and the target node of the newly created edge.

    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a graph with with m nodes and no edges.

    References
    ----------
    .. [1] A. L. Barabasi and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    np.random.seed(seed)
    G = nx.Graph()

    # 1. Init nodes
    nodes, labels, NM, Nm = _init_nodes(N,fm)
    
    G.graph = {'name':MODEL_NAME, 'label':CLASS, 'groups': GROUPS}
    G.add_nodes_from([(n, {CLASS:l}) for n,l in zip(*[nodes,labels])])
  
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
                target = random.choices(
                    list(targets_tc.keys()), # Nodes themselves
                    weights=list(targets_tc.values()), # Weight by frequency
                    k=1)[0] # Select k=1 target
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

            # Finally add edge to graph
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
    np.random.shuffle(nodes)
    majority = int(round(N*(1-fm)))
    labels = [LABELS[i >= majority] for i,n in enumerate(nodes)]
    return nodes, labels, majority, N-majority
  
def _pick_pa_h_target(
        G: nx.Graph,
        source: int,
        target_set: Set[int],
        labels: Set[bool],
        homophily: float) -> Union[int, None]:
    """Picks a random target node based on the homophily/preferential attachment dynamic.

    Args:
        G (nx.Graph): Current graph instance
        source (int): Newly added node
        target_set (Set[int]): Potential target nodes in the graph
        minority_nodes (Set[bool]): Set of minority nodes
        homophily (float): Effect of homophily.
            If high, nodes with same group membership are more likely to connect to one another.

    Returns:
        int or None: Target node that an edge should be added to
    """
    # Collect probabilities to connect to each node in target_list
    target_prob = {}
    prob_sum = 0.
    for target in target_set:
        # Effect of preferential attachment
        target_prob[target] = G.degree(target) + EPSILON
        target_prob[target] *= homophily[labels[source],labels[target]]
        
        # # Homophily effect
        # if (source in minority_nodes) ^ (target in minority_nodes):
        #     target_prob[target] *= 1 - homophily # Separate groups
        # else:
        #     target_prob[target] *= homophily # Same groups
        
        # Track sum on the fly
        prob_sum += target_prob[target]

    # Find final target by roulette wheel selection
    cum_sum = 0.
    chance = random.random()
    # Increase cumsum by node probabilities
    # High probabilities take more space
    # Respective nodes are thus more likely to be hit by the random number
    # Cumsum exceeds the selected random number if a hit occurs
    # A node with 3/5 of the total prob., will be hit in 3/5 of the cases
    for target in target_set:
        cum_sum += target_prob[target] / prob_sum
        if cum_sum > chance:
            return target
    return None

def _pick_targets(G,source,target_list,minority_mask,homophily,m):
    # Probability to connect to a target (node_id: int -> prob: float)
    target_prob_dict = {}
    for target in target_list:
        # Homophily if both nodes are from the same group, else 1 - homophily
        target_prob = homophily if minority_mask[source] == minority_mask[target] else 1 - homophily
        # Preferential attachment dynamic, i.e. likely to connect to high degree nodes
        target_prob *= (G.degree(target)+0.00001)
        target_prob_dict[target] = target_prob

    # Compute sum for normalization
    prob_sum = sum(target_prob_dict.values())
    # Return empty set if there are no targets
    if prob_sum == 0:
        return set()

    # Remember which targets were connected to
    mask_target_used = [False for _ in range(len(target_list))]
    # Counter for added edges
    count_looking = 0

    # Set of final targets
    targets = set()

    while len(targets) < m:
        count_looking += 1
        # Break if node fails to find target
        if count_looking > len(G):
            break

        # Pick target using roulette wheel selection
        rand_num = random.random()
        cumsum = 0.0
        # Increase cumsum by node probabilities
        # High probabilities take more space
        # Respective nodes are thus more likely to be hit by the random number
        # If a hit occurs, cumsum exceeds the selected random number
        # A node with 3/5 of the total prob., will be hit in 3/5 of the cases
        for j, node in enumerate(target_list):
            if mask_target_used[j]:
                continue
            cumsum += float(target_prob_dict[node]) / prob_sum
            # In case of hit, add node as target and remove for next edges
            if rand_num < cumsum:
                targets.add(node)
                mask_target_used[j] = True
                break
    return targets