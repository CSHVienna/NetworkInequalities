import copy

def G_attr_to_str(G,attr):
    G_out = copy.deepcopy(G)
    for n in G_out.nodes():
        G_out.nodes[n][attr] = str(G_out.nodes[n][attr])
    return G_out