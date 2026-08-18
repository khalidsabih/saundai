"""Training-data contracts for the professional DJ brain."""

from .curriculum import training_curriculum
from .provenance import audit_dataset_registry

__all__ = ["training_curriculum", "audit_dataset_registry"]
