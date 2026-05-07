from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineThresholds:
    """Runtime thresholds for CUP-Mem."""

    state_salience_threshold: float = 0.30
    explicit_min_confidence: float = 0.45
    strong_support_min_confidence: float = 0.55
    medium_support_min_confidence: float = 0.60
    verifier_fallback_threshold: float = 0.45
    candidate_top_k: int = 5
    latent_invalidation_temperature: float = 0.30
    invalidation_lane_max_workers: int = 4
    invalidation_merge_max_keep: int = 2
    invalidation_global_recall_top_k: int = 4
    query_top_k: int = 8
    fallback_top_k: int = 6


@dataclass
class TraceConfig:
    """Controls auxiliary trace emission for CUP-Mem."""

    enable_debug_trace: bool = False
