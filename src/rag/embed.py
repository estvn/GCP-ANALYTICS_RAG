"""multilingual-e5-small on ONNX Runtime (int8, CPU, capped threads)."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PREFIXES = {"query": "query: ", "passage": "passage: "}  # required by the E5 family
DIM = 384


class Embedder:
    def __init__(self, model_dir: Path, threads: int = 4, max_tokens: int = 512):
        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # CPU budget: ~25% of 16 threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_tokens)
        self.tok.enable_padding()

    def embed(self, texts: list[str], kind: str, batch_size: int = 16) -> np.ndarray:
        """Unit-norm vectors, shape (len(texts), 384). kind: 'query' or 'passage'."""
        if kind not in PREFIXES:
            raise ValueError(f"kind must be one of {list(PREFIXES)}")
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch([PREFIXES[kind] + t for t in texts[i : i + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            hidden = self.session.run(
                None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            )[0]
            m = mask[:, :, None]
            pooled = (hidden * m).sum(axis=1) / m.sum(axis=1)  # mean pooling over real tokens
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.vstack(out) if out else np.empty((0, DIM), dtype=np.float32)
