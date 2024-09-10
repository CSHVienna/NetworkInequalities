from typing import Dict, Any, List, Callable, Optional
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

    _event_handlers: Dict[Event, List[Callable[[Any], None]]]

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

    def remove_event_handler(
            self, event: Event,
            function: Optional[Callable[[Any], None]] = None):
        """De-registers an event handler.

        If ``function`` is provided, only the specific function is removed.
        Otherwise, all functions are deleted.

        Parameters
        ----------
        event : Event
            The :class:`.Event` to remove the handling function from.
        function : Optional[Callable[[Any], None]], optional
            The function to be removed, by default ``None``.
            If ``None``, all functions are removed.
            Otherwise, only the specified ones will be deleted.
        """
        assert event in self.EVENTS,\
            f"Event {event} not specified in {self.__class__.__name__}.__events."
        assert event in self._event_handlers,\
            f"Event {event} was not registered."
        if function is None:
            self._event_handlers[event] = []
        else:
            assert function in self._event_handlers,\
            f"Function {function} not registered for event {event}."
            self._event_handlers[event].remove(function)
