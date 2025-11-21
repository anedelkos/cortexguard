from enum import Enum


class AnomalySeverity(str, Enum):
    LOW = "low"  # Logging / Info only
    MEDIUM = "medium"  # Warning / Local retry might be needed
    HIGH = "high"  # Critical / Safety Stop required
