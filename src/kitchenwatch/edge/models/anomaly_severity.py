from enum import Enum


class AnomalySeverity(str, Enum):
    """
    Defines the standard severity levels for anomalies, used for aggregation and alerting.
    Inherits from str for easy serialization (e.g., in JSON or database).
    """

    LOW = "low"  # Logging / Info only
    MEDIUM = "medium"  # Warning / Local retry might be needed
    HIGH = "high"  # Critical / Safety Stop required


# --- SEVERITY RANKING MAPPING ---
# Maps the AnomalySeverity Enum objects to an integer for comparison checks (HIGHER number is MORE severe).
# This is the single source of truth for severity ordering.
SEVERITY_RANKING: dict[AnomalySeverity, int] = {
    AnomalySeverity.LOW: 1,
    AnomalySeverity.MEDIUM: 2,
    AnomalySeverity.HIGH: 3,
}
