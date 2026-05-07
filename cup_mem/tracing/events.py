from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TraceEvent:
    phase: str
    payload: Dict[str, Any] = field(default_factory=dict)
