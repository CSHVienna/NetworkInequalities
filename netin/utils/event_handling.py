from typing import Dict, Any, List, Callable
from collections import defaultdict
from enum import Enum

class Event(Enum):
    """Specifies events that are triggered when the specified action occurs."""

    LINK_ADD_BEFORE = "LINK_ADD_BEFORE"
    """Before an edge is added to a graph.

    :meta hide-value:"""

    LINK_ADD_AFTER = "LINK_ADD_AFTER"
    """After an edge is added to a graph.

    :meta hide-value:"""

    SIMULATION_START = "SIMULATION_START"
    """When the model simulation starts.

    :meta hide-value:"""

    SIMULATION_END = "SIMULATION_END"
    """When the model simulation terminates.

    :meta hide-value:"""

    TARGET_SELECTION_LOCAL = "LOCAL_TARGET_SELECTED"
    """When a target node in :class:`.PATCHModel` is chosen locally.

    :meta hide-value:"""

    TARGET_SELECTION_GLOBAL = "GLOBAL_TARGET_SELECTED"
    """When a target node in :class:`.PATCHModel` is chosen globally.

    :meta hide-value:"""

class HasEvents:
    """Interface to be implemented when a class triggers events.
    """

    EVENTS: List[Event] = []
    """List of implemented events.
    Should be overwritten by classes that use this interface.

    :meta hide-value:"""

    _event_handlers: Dict[Event, Callable[[Any], None]]

    def __init__(self, *args, **kwargs):
        self._event_handlers = defaultdict(list)
        super().__init__(*args, **kwargs)

    def trigger_event(self, *args, event: Event, **kwargs):
        """Triggers an event and calls all registered callback functions.
        This function should be called by the class implementing classes.

        Parameters
        ----------
        event : Event
            The event to be triggered.
        """
        assert event in self.EVENTS,\
            f"Event {event} not specified in {self.__class__.__name__}.__events."
        for function in self._event_handlers[event]:
            function(*args, **kwargs)

    def register_event_handler(
        self, event: Event, function: Callable[[Any], None]):
        """Registers a new callback function for a given event.
        This can be used to inject code which will be executed when the respective event is called.

        Parameters
        ----------
        event : Event
            The event to register the callback for.
        function : Callable[[Any], None]
            The function to be called when the event is triggered.
        """
        assert event in self.EVENTS,\
            f"Event {event} not specified in {self.__class__.__name__}.__events."
        self._event_handlers[event].append(function)
