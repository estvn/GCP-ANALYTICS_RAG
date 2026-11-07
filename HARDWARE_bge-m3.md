# Hardware cost: bge-m3

Embedding model `bge-m3` (int8 ONNX Runtime). Measured in E24 (`eval/experiments.md`): 60 real chunks, 4 threads, int8 ONNX on CPU, the same script for both
models. The CPU percentages are of 16 hardware threads. RAM includes the Python process (about 69 MB before loading
the model). One run per model, so treat the numbers as an order of magnitude.

| | value |
|---|---|
| model file | 542 MB |
| embedding dimension / tokens read per text | 1024 / 1024 (the model supports 8192; capped to bound CPU time) |
| load time | 3.8 s |
| RAM: before, after loading, peak | 69, 880, 1591 MB |
| speed on 60 chunks | 96.4 s = 0.6 chunks/s = 205 tokens/s |
| CPU, average / peak | 23.0% / 27.4% |
| query latency, median / p95 | 59 ms / 71 ms |

Selection: `load_embedder("bge-m3")`. No query or passage prefix; pooling is CLS.
It is about 9 times slower than e5-small to index and needs about 1.6 GB of RAM at peak, which still fits the project
budget because the CPU use is capped by the thread count.
