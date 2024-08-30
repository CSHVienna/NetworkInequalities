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

class CompositeLFM(enum.Enum):
    UNIFORM="UNIFORM"
    HOMOPHILY="HOMOPHILY"
    PAH="PAH"

class PATCHModel(UndirectedModel, BinaryClassModel, HasEvents):
    EVENTS = [
        Event.SIMULATION_START, Event.SIMULATION_END,
        Event.LOCAL_TARGET_SELECTION, Event.GLOBAL_TARGET_SELECTION]
    SHORT = "PATCH"

    lfm_local: CompositeLFM
    lfm_global: CompositeLFM

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
            lfm_local: CompositeLFM,
            lfm_global: CompositeLFM,
            lfm_params: Optional[Dict[str, float]] = None,
            seed:  Union[int, np.random.Generator] = 1,
            **kwargs):
        validate_float(p_tc, 0, 1)
        super().__init__(
            *args, N=N, m=m, f_m=f_m,
            seed=seed, **kwargs)
        self.p_tc = p_tc

        assert lfm_local in CompositeLFM.__members__.values(),\
            f"Invalid local link formation mechanism `{lfm_local}`"
        assert lfm_global in CompositeLFM.__members__.values(),\
            f"Invalid global link formation mechanism `{lfm_global}`"
        self.lfm_local = lfm_local
        self.lfm_global = lfm_global
        self.lfm_params = lfm_params

    def _initialize_lfms(self):
        self.tc = TriadicClosure(
            N=self._n_nodes_total,
            graph=self.graph)

        self.uniform = Uniform(N=self._n_nodes_total)
        if (self.lfm_local in (CompositeLFM.HOMOPHILY, CompositeLFM.PAH))\
            or (self.lfm_global in (CompositeLFM.HOMOPHILY, CompositeLFM.PAH)):
            assert (self.lfm_params is not None)\
                and ("h_m" in self.lfm_params)\
                and ("h_M" in self.lfm_params),\
                    "Homophily parameters must be provided"
            self.h = TwoClassHomophily.from_two_class_homophily(
                homophily=(self.lfm_params["h_M"], self.lfm_params["h_m"]),
                node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE)
            )
        if CompositeLFM.PAH in (self.lfm_local, self.lfm_global):
            self.pa = PreferentialAttachment(
                N=self._n_nodes_total,
                graph=self.graph)

    def _get_composite_target_probabilities(self, lfm: CompositeLFM, source: int) -> np.ndarray:
        p_target = self.uniform.get_target_probabilities(source)
        if lfm == CompositeLFM.HOMOPHILY:
            p_target *= self.h.get_target_probabilities(source)
        elif lfm == CompositeLFM.PAH:
            p_target *= self.pa.get_target_probabilities(source)
            p_target *= self.h.get_target_probabilities(source)
        return p_target

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        p_target = super().compute_target_probabilities(source)
        if self._rng.uniform() < self.p_tc:
            p_target *= self.tc.get_target_probabilities(source)
            p_target *= self._get_composite_target_probabilities(
                source=source, lfm=self.lfm_local)
            self.trigger_event(event=Event.LOCAL_TARGET_SELECTION)
        else:
            p_target *= self._get_composite_target_probabilities(
                source=source, lfm=self.lfm_global)
            self.trigger_event(event=Event.GLOBAL_TARGET_SELECTION)

        return p_target / p_target.sum()
