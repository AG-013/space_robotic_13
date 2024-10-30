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



# class PlannerType(Enum):
#     ERROR                    = 0
#     WAITING_FOR_MAP          = 1
#     SELECTING_FRONTIER       = 2
#     MOVING_TO_FRONTIER       = 3
#     EXPLORATION              = 4
#     HANDLE_REJECTED_FRONTIER = 5
#     HANDLE_TIMEOUT           = 6
#     OBJECT_IDENTIFIED_SCAN   = 7
#     EXPLORED_MAP             = 8
#     IDENTIFYING_FRONTIERS    = 9



# class PlannerType(Enum):
#     ERROR = 0
#     MOVE_FORWARDS = 1
#     RETURN_HOME = 2
#     GO_TO_FIRST_ARTIFACT = 3
#     RANDOM_WALK = 4
#     RANDOM_GOAL = 5
#     EXPLORATION = 6
    
    
# class ExplorationsState(Enum):
#     ERROR = 0
#     WAITING_FOR_MAP = 1
#     IDENTIFYING_FRONTIERS = 2
#     SELECTING_FRONTIER = 3
#     MOVING_TO_FRONTIER = 4
#     HANDLE_REACHED_FRONTIER = 5
#     HANDLE_REJECTED_FRONTIER = 6
#     HANDLE_TIMEOUT = 7
#     OBJECT_IDENTIFIED_SCAN = 8
#     EXPLORED_MAP = 9