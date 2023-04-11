from netin.utils import constants as const
from .digraph import DiGraph


class DPA(DiGraph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, d: float, f_m: float, plo_M: float, plo_m: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        d: float
            edge density (minimum=0, maximum=1)

        plo_M: float
            activity (out-degree power law exponent) majority group (minimum=1)

        plo_m: float
            activity (out-degree power law exponent) minority group (minimum=1)

        seed: object
            seed for random number generator

        Notes
        -----
        The initialization is a digraph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        super().__init__(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.DPA_MODEL_NAME)
