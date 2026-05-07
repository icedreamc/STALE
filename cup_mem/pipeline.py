from __future__ import annotations

from typing import List, Optional

from .retrieval.embedding import BaseRetriever, build_retriever
from .llm_layer.client import LLMClient
from .memory.models import (
    SessionChunk,
    SessionDelta,
)
from .memory.schema import schema_prompt_text
from .core.config import PipelineThresholds, TraceConfig
from .core.engine_utils import EngineUtilityMixin
from .core.sample_runner import SampleRunnerMixin
from .store_layer import ProfileStore
from .write.chunker import ChunkerMixin
from .write.delta_extractor import DeltaExtractorMixin
from .write.update_resolver import UpdateResolverMixin
from .write.invalidation_resolver import InvalidationResolverMixin
from .write.stale_linker import StaleLinkerMixin
from .write.writer import SessionWriterMixin
from .query.action_grounding import ActionGroundingMixin
from .query.engine import QueryEngineMixin
from .query.retrieval import QueryRetrievalMixin
from .query.premise_verifier import VerifierBasisMixin


class CupMemEngine(EngineUtilityMixin, ChunkerMixin, DeltaExtractorMixin, UpdateResolverMixin, InvalidationResolverMixin, StaleLinkerMixin, SessionWriterMixin, VerifierBasisMixin, ActionGroundingMixin, QueryRetrievalMixin, QueryEngineMixin, SampleRunnerMixin):
    def __init__(
        self,
        *,
        llm: LLMClient,
        embedding_model_path: Optional[str] = None,
        embedding_device: str = "cpu",
        thresholds: Optional[PipelineThresholds] = None,
        trace_config: Optional[TraceConfig] = None,
    ):
        self.llm = llm
        self.thresholds = thresholds or PipelineThresholds()
        self.trace_config = trace_config or TraceConfig()
        if not embedding_model_path:
            raise ValueError(
                "embedding_model_path is required. Please pass a local sentence-embedding model directory "
                "(for example, a local all-MiniLM-L6-v2 checkpoint)."
            )
        self.store = ProfileStore()
        self.embedding: BaseRetriever = build_retriever(
            embedding_model_path,
            device=embedding_device,
        )
        self.chunk_bank: List[SessionChunk] = []
        self.delta_store: List[SessionDelta] = []
        self._chunk_counter = 0
        self._delta_counter = 0
        self._proposal_counter = 0
        self.schema_text = schema_prompt_text()

    def reset(self) -> None:
        self.store.reset()
        self.chunk_bank = []
        self.delta_store = []
        self._chunk_counter = 0
        self._delta_counter = 0
        self._proposal_counter = 0

    def _next_chunk_id(self) -> str:
        self._chunk_counter += 1
        return f"c_{self._chunk_counter:05d}"

    def _next_delta_id(self) -> str:
        self._delta_counter += 1
        return f"d_{self._delta_counter:05d}"

    def _next_proposal_id(self) -> str:
        self._proposal_counter += 1
        return f"lip_{self._proposal_counter:05d}"
