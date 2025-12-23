# Retriever Evaluation

This document contains the results of retriever evaluation experiments.

## K Sweep

Evaluation of retriever performance across different K values using a fixed embedding model.

| Parameter | Hit@K |
|-----------|-------|
| 3 | 0.451 |
| 5 | 0.488 |
| 7 | 0.524 |
| 9 | 0.561 |

**Conclusion:**

The best configuration is **9** with **Hit@K = 0.561**.

## Embedding Model Comparison

Comparison of embedding models at fixed K = 9.

| Parameter | Hit@K |
|-----------|-------|
| cointegrated/rubert-tiny2 | 0.561 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.268 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.683 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 | 0.683 |
| sentence-transformers/LaBSE | 0.756 |

**Conclusion:**

The best configuration is **sentence-transformers/LaBSE** with **Hit@K = 0.756**.

## Chunking Strategy Comparison

Comparison of different chunk size and overlap configurations.

| Parameter | Hit@K |
|-----------|-------|
| c400_o50 | 0.744 |
| c600_o100 | 0.756 |
| c800_o150 | 0.78 |
| c1000_o200 | 0.817 |

**Conclusion:**

The best configuration is **c1000_o200** with **Hit@K = 0.817**.

