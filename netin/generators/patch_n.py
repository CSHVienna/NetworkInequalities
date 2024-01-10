from typing import Union, Dict, Any, List, Tuple, Type
from collections import defaultdict
from enum import Enum
from .h import Homophily
from .pah import PAH

import numpy as np

from netin.utils import constants as const
from netin.generators.undirected import UnDiGraph

class LINK_FORMATION_MECHANISMS(Enum):
    H="H"
    PAH="PAH"

class NewPATCH(PAH):
    _node_source_curr: int
    _tc_candidates: Dict[int, int]
    _g_local: Type[UnDiGraph]
    _g_global: Type[UnDiGraph]


    lfm_local: str
    lfm_global: str
    tc: float

    def __init__(self,
                 n: int, k: int, f_m: float, h_mm: float, h_MM: float, tc: float,
                 lfm_local: LINK_FORMATION_MECHANISMS, lfm_global: LINK_FORMATION_MECHANISMS,
                 seed: object = None,
                 **kwargs):
        PAH.__init__(self, n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        self.tc = tc
        self.lfm_local = lfm_local
        self.lfm_global = lfm_global
        self._g_local = Homophily if lfm_local == LINK_FORMATION_MECHANISMS.H else PAH
        self._g_global = Homophily if lfm_global == LINK_FORMATION_MECHANISMS.H else PAH
        self._node_source_curr = -1
        self.init_special_targets(self._node_source_curr)
        self.model_name = const.PATCH_MODEL_NAME

    def get_metadata_as_dict(self) -> Dict[str, Any]:
        """
        Returns the metadata (parameters) of the model as a dictionary.

        Returns
        -------
        dict
            metadata of the model
        """
        obj = Homophily.get_metadata_as_dict(self)
        obj.update(PAH.get_metadata_as_dict(self))
        obj.update({
            'tc': self.tc,
            "lfm_local": self.lfm_local,
            "lfm_global": self.lfm_global,
        })
        return obj

    def init_special_targets(self, source: int) -> object:
        """
        Returns an empty dictionary (source node ids)

        Parameters
        ----------
        source : int
            Newly added node

        Returns
        -------
        object
            Return an empty dictionary (source node ids)
        """
        self._node_source_curr = source
        self._tc_candidates = defaultdict(int)

    def get_target_probabilities(self, source: int, available_nodes: List[int]) -> Tuple[np.array, List[int]]:
        """
        Returns the probabilities of nodes to be selected as target nodes.

        Parameters
        ----------
        source: int
            source node id

        available_nodes: List[int]
            list of available nodes to connect to

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            probabilities of nodes to be selected as target nodes, and list of target of nodes

        """
        tc_prob = np.random.random()

        if source != self._node_source_curr:
            self.init_special_targets(source)

        if tc_prob < self.tc and len(self._tc_candidates) > 0:
            target_nodes = []
            for target, cnt in self._tc_candidates.items():
                for _ in range(cnt):
                    target_nodes.append(target)
            return self._g_local.get_target_probabilities(
                self,
                source=source, available_nodes=target_nodes
                # source=source, available_nodes=list(self._tc_candidates.keys())
            )
        return self._g_global.get_target_probabilities(
            self,
            source=source, available_nodes=available_nodes)

    def on_edge_added(self, source: int, target: int):
        """
        Updates the set of special available_nodes based on the triadic closure mechanism.
        When an edge is created, multiple potential triadic closures emerge (i.e., two-hop neighbors that are not yet
        directly connected). These are added to the set of special available_nodes.

        Parameters
        ----------
        idx_target: int
            index of the target node

        source: int
            source node

        target: int
            target node

        available_nodes: List[int]
            list of target nodes

        special_targets: Union[None, Dict[int, int]]
            special available_nodes

        Returns
        -------
        Union[None, Dict[int, int]
            updated special available_nodes
        """
        if target in self._tc_candidates:
            del self._tc_candidates[target]
        for neighbor in self.neighbors(target):
            # G[source] gives direct access (O(1)) to source's neighbors
            # G.neighbors(source) returns an iterator which would
            # need to be searched iteratively
            if neighbor not in self[source]:
                self._tc_candidates[neighbor] += 1
        return super().on_edge_added(source, target)

    def info_params(self):
        """
        Shows the (input) parameters of the graph.
        """
        self._g_local.info_params(self)
        self._g_global.info_params(self)

    def info_computed(self):
        """
        Shows the (computed) properties of the graph.
        """
        self._g_local.info_computed(self)
        self._g_global.info_computed(self)

    def infer_triadic_closure(self) -> float:
        """
        Infers analytically the triadic closure value of the graph.
        @TODO: This still needs to be implemented.
        Returns
        -------
        float
            triadic closure probability of the graph
        """
        raise NotImplementedError("Inferring triadic closure probability not implemented yet.")

    def makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              tc=self.tc,
                              h_mm=self.h_mm,
                              h_MM=self.h_MM,
                              lfm_local=self.lfm_local,
                              lfm_global=self.lfm_global,
                              seed=self.seed)
