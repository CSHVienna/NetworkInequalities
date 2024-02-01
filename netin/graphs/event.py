from enum import Enum

class Event(Enum):
    """Specification of events"""
    LINK_ADD_BEFORE = "LINK_ADD_BEFORE"
    LINK_ADD_AFTER = "LINK_ADD_AFTER"
