from typing import Dict, Any, List, Callable

from enum import Enum

class Event(Enum):
    """Specification of events"""
    LINK_ADD_BEFORE = "LINK_ADD_BEFORE"
    LINK_ADD_AFTER = "LINK_ADD_AFTER"

    SIMULATION_START = "SIMULATION_START"
    SIMULATION_END = "SIMULATION_END"

class HasEvents:
    EVENTS: List[Event] = []
    _event_handlers: Dict[Event, Callable[[Any], None]] = {}

    def trigger_event(self, *args, event: Event, **kwargs):
        assert event in self.EVENTS,\
            f"Event {event} not specified in {self.__class__.__name__}.__events."
        for function in self._event_handlers[event]:
            function(*args, **kwargs)

    def register_event_handler(
        self, event: Event, function: Callable[[Any], None]):
        assert event in self.EVENTS,\
            f"Event {event} not specified in {self.__class__.__name__}.__events."
        self._event_handlers[event].append(function)