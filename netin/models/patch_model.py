from typing import Union, Optional
import enum

import numpy as np

from .undirected_model import UndirectedModel
from .binary_class_model import BinaryClassModel
from ..utils.event_handling import Event
from ..utils.constants import CLASS_ATTRIBUTE
from ..utils.validator import validate_float
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment
from ..link_formation_mechanisms.triadic_closure import TriadicClosure
from ..link_formation_mechanisms.uniform import Uniform

class CompoundLFM(enum.Enum):
    """A combination of link formation mechanism.

    This class is used to define how triadic closure or global links
    should be formed in the :class:`.PATCHModel`.

    :meta hide-value:

    Attributes
    ----------
    UNIFORM : str
        Targets are chosen uniformly at random.
    HOMOPHILY : str
        Targets are chosen based on homophily (see :class:`.Homophily`
        and :class:`.HomophilyModel` for details).
    PAH : str
        Targets are chosen based on homophily and preferential attachment
        (see :class:`.PAHModel` for details).
    """

    UNIFORM="UNIFORM"
    HOMOPHILY="HOMOPHILY"
    PAH="PAH"

class PATCHModel(
    UndirectedModel, BinaryClassModel):
    """The PATCHModel joins nodes to the network based on a combination of
    [P]referential [A]ttachment, [T]riadic [C]losure and [H]omophily.
    Based on the triadic closure probability :attr:`tau`, links are formed either
    globally (1-:attr:`tau`) or among neighbors of existing neighbors (:attr:`tau`).
    Globally, nodes can connect to any other node in the network.

    How a target node is selected from the set of available nodes then
    depends on the other link formation mechanisms of preferential attachment and/or homophily.
    See :attr:`.lfm_tc` and :attr:`lfm_global` for details.

    Parameters
    ----------
    n : int
        The number of nodes to be added.
    f_m : float
        The fraction of the minority group.
    k : int
        The number of new edges per node.
    tau : float
        The probability for triadic closure, meaning that an edge will
        be formed locally among the neighbors of existing neighbors.
        With the complementary probability (``1 - tau``), all existing
        nodes are available for connection.
        See :attr:`lfm_tc` and :attr:`lfm_global` for a specification of how
        targets are chosen from either set.
    lfm_tc : CompoundLFM
        Defines how triadic closure targets are chosen.
        Both :attr:`lfm_tc` and :attr:`lfm_global` can be set to any value
        defined in :class:`.CompoundLFM`:

        1. :attr:`.CompoundLFM.UNIFORM`: the target nodes are chosen randomly
        2. :attr:`.CompoundLFM.HOMOPHILY`: the target nodes are chosen based on homophily
        3. :attr:`.CompoundLFM.PAH`: the target nodes are chosen based on preferential attachment
            and homophily (choose ``h_mm = h_MM = 0.5`` to neutralize the effect of homophily;
            see :class:`.PAHModel` for details).

        For options 2. and 3. the ``h_mm`` and ``h_MM`` parameters must be provided
        to specify the homophily values of the minority and
        majority groups respectively.
    lfm_global : CompoundLFM
        Defines how global targets are chosen.
        See :attr:`lfm_tc` for details.
    h_mm : Optional[float], optional
        Homophily parameter for minority nodes, by default None.
        If either triadic closure or global link formation mechanisms contains
        homophily (:attr:`.CompoundLFM.HOMOPHILY` or :attr:`.CompoundLFM.PAH`),
        this parameter must be provided.
        See :class:`.HomophilyModel` for details on the homophily parameters.
    h_MM : Optional[float], optional
        Homophily parameter for majority nodes, by default None.
        If either triadic closure or global link formation mechanisms contains
        homophily (:attr:`.CompoundLFM.HOMOPHILY` or :attr:`.CompoundLFM.PAH`),
        this parameter must be provided.
        See :class:`.HomophilyModel` for details on the homophily parameters.
    seed : Union[int, np.random.Generator], optional
        Random seed or random number generator, by default None
    """

    EVENTS = [
        Event.TARGET_SELECTION_LOCAL, Event.TARGET_SELECTION_GLOBAL] + UndirectedModel.EVENTS
    SHORT = "PATCH"

    lfm_tc: CompoundLFM
    lfm_global: CompoundLFM

    tau: float
    h_MM: Optional[float]
    h_mm: Optional[float]

    uniform: Uniform
    tc: TriadicClosure
    h: Optional[TwoClassHomophily]
    pa: Optional[PreferentialAttachment]

    _node_curr: int

    def __init__(
            self, *args,
            n: int, f_m: float, k:int,
            tau: float,
            lfm_tc: CompoundLFM,
            lfm_global: CompoundLFM,
            h_MM: Optional[float] = None,
            h_mm: Optional[float] = None,
            seed:  Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        validate_float(tau, 0, 1)
        super().__init__(
            *args, n=n, k=k, f_m=f_m,
            seed=seed, **kwargs)
        self.tau = tau

        assert lfm_tc in CompoundLFM.__members__.values(),\
            f"Invalid triadic closure link formation mechanism `{lfm_tc}`"
        assert lfm_global in CompoundLFM.__members__.values(),\
            f"Invalid global link formation mechanism `{lfm_global}`"
        self.lfm_tc = lfm_tc
        self.lfm_global = lfm_global
        self._node_curr = -1

        if lfm_tc in (CompoundLFM.HOMOPHILY, CompoundLFM.PAH)\
            or lfm_global in (CompoundLFM.HOMOPHILY, CompoundLFM.PAH):
            assert None not in (h_MM, h_mm), "Homophily parameters must be provided"
            self.h_mm = h_mm
            self.h_MM = h_MM

    def _initialize_lfms(self):
        """Initializes and configures the link formation mechanisms.
        This depends on the choice of :attr:`lfm_tc` and :attr:`lfm_global`.
        The parameters are given by ``lfm_params``.
        """
        self.tc = TriadicClosure(
            n=self._n_nodes_total,
            graph=self.graph)
        self.uniform = Uniform(n=self._n_nodes_total)

        if (self.lfm_tc in (CompoundLFM.HOMOPHILY, CompoundLFM.PAH))\
            or (self.lfm_global in (CompoundLFM.HOMOPHILY, CompoundLFM.PAH)):

            assert self.h_MM is not None and self.h_mm is not None, \
                "Homophily parameters must be provided when using homophily-based LFMs"
            self.h = TwoClassHomophily.from_two_class_homophily(
                homophily=(self.h_MM, self.h_mm),
                node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE)
            )
        if CompoundLFM.PAH in (self.lfm_tc, self.lfm_global):
            self.pa = PreferentialAttachment(
                n=self._n_nodes_total,
                graph=self.graph)

    def _get_compound_target_probabilities(self, lfm: CompoundLFM, source: int)\
        -> np.ndarray:
        """Return the compound link formation mechanism probability.

        Returns
        -------
        numpy.ndarray
            The target probabilities depending on the chosen :class:`.CompoundLFM`.
        """
        if lfm == CompoundLFM.HOMOPHILY:
            assert self.h is not None, "Homophily LFM not initialized"
            return np.array(self.h.get_target_probabilities(source))
        if lfm == CompoundLFM.PAH:
            assert self.pa is not None, "Preferential Attachment LFM not initialized"
            assert self.h is not None, "Homophily LFM not initialized"
            return np.array(self.pa.get_target_probabilities(source))\
                * np.array(self.h.get_target_probabilities(source))
        return np.array(self.uniform.get_target_probabilities(source))

    def _get_tc_target_probabilities(self, source: int) -> np.ndarray:
        self.trigger_event(event=Event.TARGET_SELECTION_LOCAL, source=source)
        return self._get_compound_target_probabilities(
            source=source, lfm=self.lfm_tc)

    def _get_global_target_probabilities(self, source: int) -> np.ndarray:
        self.trigger_event(event=Event.TARGET_SELECTION_GLOBAL, source=source)
        return self._get_compound_target_probabilities(
            source=source, lfm=self.lfm_global)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Compute the target probabilities based on triadic closure and
        the specified compound link formation mechanisms for global and
        triadic closure links.

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

        if self._node_curr != source:
            self._node_curr = source
            p_target *= self._get_global_target_probabilities(source)
        else:
            if self._rng.uniform() < self.tau:
                p_target *= self.tc.get_target_probabilities(source)
                p_target *= self._get_tc_target_probabilities(source)
            else:
                p_target *= self._get_global_target_probabilities(source)

        return p_target / p_target.sum()
