from typing import Union, Optional, Dict
import enum

import numpy as np

from .undirected_model import UndirectedModel
from .binary_class_model import BinaryClassModel
from ..utils.event_handling import Event, HasEvents
from ..utils.constants import CLASS_ATTRIBUTE
from ..utils.validator import validate_float
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment
from ..link_formation_mechanisms.triadic_closure import TriadicClosure
from ..link_formation_mechanisms.uniform import Uniform

class CompoundLFM(enum.Enum):
    """A combination of link formation mechanism.
    This class is used to define how local or global links should be formed in the `PATCHModel`.
    """
    UNIFORM="UNIFORM"
    """Targets are chosen uniformly at random.

    :meta hide-value:"""

    HOMOPHILY="HOMOPHILY"
    """Targets are chosen based on homophily (see :class:`.Homophily`
       and :class:`.HomophilyModel` for details).

    :meta hide-value:"""

    PAH="PAH"
    """Targets are chosen based on homophily and preferential attachment
       (see :class:`.PAHModel` for details).

    :meta hide-value:"""

class PATCHModel(
    UndirectedModel, BinaryClassModel, HasEvents):
    """The PATCHModel joins nodes to the network based on a combination of
    [P]referential [A]ttachment, [T]riadic [C]losure and [H]omophily.
    Based on the triadic closure probability `p_tc`, links are formed either locally or globally.
    For local links, nodes can connect only to neighbors of existing neighbors.
    Globally, nodes can connect to any other node in the network.

    How a target node is selected from the set of available nodes then
    depends on the other link formation mechanisms of preferential attachment and/or homophily.
    See :attr:`.lfm_local` and :attr:`lfm_global` for details.

    Parameters
    ----------
    N : int
        The number of nodes to be added.
    f_m : float
        The fraction of the minority group.
    m : int
        The number of new edges per node.
    p_tc : float
        The probability for triadic closure, meaning that an edge will
        be formed locally among the neighbors of existing neighbors.
        With the complementary probability (1 - `p_tc`), all existing
        nodes are available for connection.
        See `lfm_local` and `lfm_global` for a specification of how
        targets are chosen from either set.
    lfm_local : CompoundLFM
        Defines how local targets are chosen.
        Both :attr:`lfm_local` and :attr:`lfm_global` can be set to any value defined in :class:`.CompoundLFM`:

        1. :attr:`.CompoundLFM.UNIFORM`: the target nodes are chosen randomly
        2. :attr:`.CompoundLFM.HOMOPHILY`: the target nodes are chosen based on homophily
        3. :attr:`.CompoundLFM.PAH`: the target nodes are chosen based on preferential attachment and homophily (choose `h_m = h_M = 0.5` to neutralize the effect of homophily; see :class:`.PAHModel` for details).

        For options 2. and 3. the `lfm_params` dictionary has
        to contain the homophily values of the minority and
        majority group (for instance by setting `lfm_params={"h_m": 0.2, "h_M": 0.8}`).
    lfm_global : CompoundLFM
        Defines how global targets are chosen.
        See `lfm_local` for details.
    lfm_params : Optional[Dict[str, float]], optional
        Dictionary containing additional parameterization of link
        formation mechanisms, by default None.
        If either local or global link formation mechanisms contains
        homophily (`CompoundLFM.Homophily` or 'CompoundLFM.PAH`), the
        dictionary should contain the keys `h_m` and `h_M`, containing
        the desired homophily parameters.
        See :class:`.HomophilyModel` for details on the homophily parameters.
    seed : Union[int, np.random.Generator], optional
        _description_, by default 1
    """

    EVENTS = [
        Event.SIMULATION_START, Event.SIMULATION_END,
        Event.TARGET_SELECTION_LOCAL, Event.TARGET_SELECTION_GLOBAL]
    SHORT = "PATCH"

    lfm_local: CompoundLFM
    lfm_global: CompoundLFM

    p_tc: float

    lfm_params: Optional[Dict[str, float]]
    uniform: Uniform
    tc: TriadicClosure
    h: Optional[TwoClassHomophily]
    pa: Optional[PreferentialAttachment]

    def __init__(
            self, *args,
            N: int, f_m: float, m:int,
            p_tc: float,
            lfm_local: CompoundLFM,
            lfm_global: CompoundLFM,
            lfm_params: Optional[Dict[str, float]] = None,
            seed:  Union[int, np.random.Generator] = 1,
            **kwargs):
        validate_float(p_tc, 0, 1)
        super().__init__(
            *args, N=N, m=m, f_m=f_m,
            seed=seed, **kwargs)
        self.p_tc = p_tc

        assert lfm_local in CompoundLFM.__members__.values(),\
            f"Invalid local link formation mechanism `{lfm_local}`"
        assert lfm_global in CompoundLFM.__members__.values(),\
            f"Invalid global link formation mechanism `{lfm_global}`"
        self.lfm_local = lfm_local
        self.lfm_global = lfm_global
        self.lfm_params = lfm_params

    def _initialize_lfms(self):
        """Initializes and configures the link formation mechanisms.
        This depends on the choice of `lfm_local` and `lfm_global`.
        The parameters are given by `lfm_params`.
        """
        self.tc = TriadicClosure(
            N=self._n_nodes_total,
            graph=self.graph)
        self.uniform = Uniform(N=self._n_nodes_total)

        if (self.lfm_local in (CompoundLFM.HOMOPHILY, CompoundLFM.PAH))\
            or (self.lfm_global in (CompoundLFM.HOMOPHILY, CompoundLFM.PAH)):
            assert (self.lfm_params is not None)\
                and ("h_m" in self.lfm_params)\
                and ("h_M" in self.lfm_params),\
                    "Homophily parameters must be provided"
            self.h = TwoClassHomophily.from_two_class_homophily(
                homophily=(self.lfm_params["h_M"], self.lfm_params["h_m"]),
                node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE)
            )
        if CompoundLFM.PAH in (self.lfm_local, self.lfm_global):
            self.pa = PreferentialAttachment(
                N=self._n_nodes_total,
                graph=self.graph)

    def _get_compound_target_probabilities(self, lfm: CompoundLFM, source: int)\
        -> np.ndarray:
        """Return the compound link formation mechanism probability.

        Returns
        -------
        numpy.ndarray
            The target probabilities depending on the chosen `CompoundLFM`.
        """
        if lfm == CompoundLFM.HOMOPHILY:
            return self.h.get_target_probabilities(source)
        if lfm == CompoundLFM.PAH:
            return self.pa.get_target_probabilities(source)\
                * self.h.get_target_probabilities(source)
        return self.uniform.get_target_probabilities(source)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Compute the target probabilities based on triadic closure and
        the specified compound link formation mechanisms for global and
        local links.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        np.ndarray
            Target probabilities for all nodes in the network.
        """
        p_target = super().compute_target_probabilities(source)
        if self._rng.uniform() < self.p_tc:
            p_target *= self.tc.get_target_probabilities(source)
            p_target *= self._get_compound_target_probabilities(
                source=source, lfm=self.lfm_local)
            self.trigger_event(event=Event.TARGET_SELECTION_LOCAL)
        else:
            p_target *= self._get_compound_target_probabilities(
                source=source, lfm=self.lfm_global)
            self.trigger_event(event=Event.TARGET_SELECTION_GLOBAL)

        return p_target / p_target.sum()
