from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from .event import Event

class BaseClass:
    EVENTS: List[Event] = []
    _event_handlers: Dict[Event, Callable[[Any], None]]
    _verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self._created_at = datetime.now()
        self._verbose = verbose

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            self.__class__.__name__: {
                'created_at': self._created_at,
                "event_handlers" : {
                event: [f.__name__ for f in functions]\
                    for event, functions in self._event_handlers.items()}
            }
        } if d_meta_data is None else d_meta_data

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
