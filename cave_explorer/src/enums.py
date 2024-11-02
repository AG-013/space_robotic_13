from enum import Enum


class PlannerType(Enum):
    ERROR                    = 0
    WAITING_FOR_MAP          = 1
    SELECTING_FRONTIER       = 2
    MOVING_TO_FRONTIER       = 3
    HANDLE_REJECTED_FRONTIER = 4
    HANDLE_TIMEOUT           = 5
    OBJECT_IDENTIFIED_SCAN   = 6
    EXPLORED_MAP             = 7