from .planner import Planner, Task

__all__ = ["Planner", "Task", "SchedulePool"]


def __getattr__(name):
    # Keep backward compatibility for:
    #   from t2v_flow.planner import SchedulePool
    # while avoiding importing OR-Tools at package import time.
    if name == "SchedulePool":
        from .schedule_pool import SchedulePool

        return SchedulePool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
