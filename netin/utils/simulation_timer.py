import time
from typing import Optional

from .event_handling import Event
from ..models.model import Model

class SimulationTimer:
    """Measures the simulation time of a :class:`.Model` execution.

    Parameters
    ----------
    model : Model
        The model to measure the time for.
    """
    _start_time: float
    time: Optional[float] = None

    def __init__(self, model: Model) -> None:
        model.register_event_handler(
            Event.SIMULATION_START, self._start_timer)
        model.register_event_handler(
            Event.SIMULATION_END, self._end_timer)

    def _start_timer(self):
        self._start_time = time.time()

    def _end_timer(self):
        self.time = time.time() - self._start_time

    def get_time(self) -> float:
        """Returns the passed simulation time in second.

        Returns
        -------
        float
            Simulation time in seconds.
        """
        assert self.time is not None,\
            "The simulation has not run yet."
        return self.time
