from enum import Enum


class PlannerType(Enum):
    ERROR                    = 0
    WAITING_FOR_MAP          = 1
    SELECTING_FRONTIER       = 2
    MOVING_TO_FRONTIER       = 3
    EXPLORATION              = 4
    HANDLE_REJECTED_FRONTIER = 5
    HANDLE_TIMEOUT           = 6
    OBJECT_IDENTIFIED_SCAN   = 7
    EXPLORED_MAP             = 8