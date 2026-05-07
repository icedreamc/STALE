from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional



class BaseRetriever:
    def rank(
        self,
        *,
        query_text: str,
        candidates: Iterable[Any],
        text_getter: Callable[[Any], str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class TransformerRetriever(BaseRetriever):
    def __init__(self, model_path: Path, device: str = "cpu"):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._F = F
        self.device = device
        self.model_path = str(model_path.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        self.model.eval()
        if any(getattr(param, "is_meta", False) for param in self.model.parameters()):
            self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            self.model.eval()
            if any(getattr(param, "is_meta", False) for param in self.model.parameters()):
                raise RuntimeError(f"Embedding model loaded with meta tensors: {self.model_path}")
        self.model.to(self.device)

    def _encode(self, texts: List[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self._torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            embeddings = self._F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def rank(
        self,
        *,
        query_text: str,
        candidates: Iterable[Any],
        text_getter: Callable[[Any], str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        candidates = list(candidates)
        if not candidates:
            return []
        texts = [query_text] + [text_getter(candidate) for candidate in candidates]
        embeddings = self._encode(texts)
        query_emb = embeddings[0:1]
        cand_embs = embeddings[1:]
        sims = self._torch.mm(query_emb, cand_embs.T).squeeze(0)
        ranked = [
            {
                "score": float(sims[idx].item()),
                "item": candidates[idx],
            }
            for idx in range(len(candidates))
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]


def build_retriever(model_path: Optional[str], device: str = "cpu") -> BaseRetriever:
    if not model_path:
        raise ValueError("embedding_model_path is required; lexical retrieval fallback is disabled.")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding model path not found: {path}")
    return TransformerRetriever(path, device=device)
