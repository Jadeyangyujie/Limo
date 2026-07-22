"""Root-conditioned geometric reachability teachers for robot-centric maps."""

from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import (
    ReachabilityState,
    TeacherAResult,
    build_teacher_a,
)

__all__ = [
    "MapGeometry",
    "ReachabilityState",
    "TeacherAResult",
    "build_teacher_a",
]
