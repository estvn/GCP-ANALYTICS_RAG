"""ONNX Runtime embedders (int8, CPU, capped threads): multilingual-e5-small and bge-m3."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass(frozen=True)
class ModelSpec:
    onnx_file: str
    max_tokens: int  # tokens read per text; longer texts are truncated
    dim: int  # length of each vector
    pooling: str  # "mean" over the real tokens, or "cls" (the first token)
    query_prefix: str
    passage_prefix: str
    token_type_ids: bool  # whether the ONNX graph takes this input


# The E5 family requires the "query: " / "passage: " prefixes.
E5_SMALL = ModelSpec("onnx/model_quantized.onnx", 512, 384, "mean", "query: ", "passage: ", True)
# bge-m3 reads up to 8192 tokens; capped at 1024 to bound CPU time (the course chunks are below 1000).
BGE_M3 = ModelSpec("onnx/model_int8.onnx", 1024, 1024, "cls", "", "", False)

MODELS = {
    "e5-small": (E5_SMALL, Path("models/multilingual-e5-small")),
    "bge-m3": (BGE_M3, Path("models/bge-m3")),
}


def pool(hidden: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """Collapse per-token vectors (batch, tokens, dim) into one unit-norm vector per text."""
    if how == "cls":
        vectors = hidden[:, 0]
    else:
        weights = mask[:, :, None]
        vectors = (hidden * weights).sum(axis=1) / weights.sum(axis=1)  # padding tokens do not count
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Embedder:
    def __init__(self, model_dir: Path, spec: ModelSpec = E5_SMALL, threads: int = 4):
        model_dir = Path(model_dir)
        self.spec = spec
        self.dim = spec.dim
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # CPU budget: ~25% of 16 threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / spec.onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(spec.max_tokens)
        self.tok.enable_padding()
        self._counter = Tokenizer.from_file(str(model_dir / "tokenizer.json"))  # no truncation: counts the real size
        self._prefixes = {"query": spec.query_prefix, "passage": spec.passage_prefix}

    def count_tokens(self, text: str) -> int:
        return len(self._counter.encode(text).ids)

    def embed(self, texts: list[str], kind: str, batch_size: int = 16) -> np.ndarray:
        """Unit-norm vectors, shape (len(texts), dim). kind: 'query' or 'passage'."""
        if kind not in self._prefixes:
            raise ValueError(f"kind must be one of {list(self._prefixes)}")
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch([self._prefixes[kind] + t for t in texts[i : i + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if self.spec.token_type_ids:
                feed["token_type_ids"] = np.zeros_like(ids)
            out.append(pool(self.session.run(None, feed)[0], mask, self.spec.pooling))
        return np.vstack(out) if out else np.empty((0, self.dim), dtype=np.float32)


def load_embedder(name: str, threads: int = 4) -> Embedder:
    spec, model_dir = MODELS[name]
    return Embedder(model_dir, spec, threads)
