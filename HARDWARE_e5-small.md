# Hardware cost: multilingual-e5-small

Embedding model `e5-small` (`multilingual-e5-small`, int8 ONNX Runtime). Measured in E24 (`eval/experiments.md`): 60 real chunks, 4 threads, int8 ONNX on CPU, the same script for both
models. The CPU percentages are of 16 hardware threads. RAM includes the Python process (about 69 MB before loading
the model). One run per model, so treat the numbers as an order of magnitude.

| | value |
|---|---|
| model file | 113 MB |
| embedding dimension / tokens read per text | 384 / 512 (longer texts are truncated) |
| load time | 0.8 s |
| RAM: before, after loading, peak | 69, 444, 912 MB |
| speed on 60 chunks | 10.6 s = 5.6 chunks/s = 1737 tokens/s |
| CPU, average / peak | 22.1% / 27.0% |
| query latency, median / p95 | 12 ms / 18 ms |

Selection: `load_embedder("e5-small")`. The model requires the prefixes `query: ` and `passage: `.
Indexing the whole course took about 2 minutes with the structural chunker (131 s measured).
