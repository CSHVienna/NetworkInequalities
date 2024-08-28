import time
from typing import Optional

from .event_handling import Event
from ..models.model import Model

class SimulationTimer:
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
