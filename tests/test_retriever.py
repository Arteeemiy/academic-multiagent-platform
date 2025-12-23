# =====================
# Imports
# =====================
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import json
from pathlib import Path
from datetime import datetime
from utils.paths import normalize_name


from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# =====================
# Experiment Configuration
# =====================

K_VALUES = [3, 5, 7, 9]

EMBEDDER_NAMES = [
    "cointegrated/rubert-tiny2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/LaBSE",
]

VECTOR_STORE_NAME = "Chroma"

RESULTS_PATH = Path("results/retrieval_eval.json")
DOCS_PATH = Path("docs/retrieval_eval.md")


# =====================
# Data Loading
# =====================

with open("tests/golden_questions.json", encoding="utf-8") as f:
    golden_data = json.load(f)


# =====================
# Vector Store Factory
# =====================


def build_vector_db(embedder_name: str) -> Chroma:
    """
    Connect to an existing Chroma vector store.
    Assumes the vector database has already been built offline.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedder_name, model_kwargs={"device": "cpu"}
    )

    return Chroma(
        collection_name="mng_data_collection",
        embedding_function=embeddings,
        persist_directory=f"./knowledge/chromadb_{normalize_name(embedder_name)}/",
    )


# =====================
# Retrieval Metrics
# =====================


def compute_hit_at_k(vector_db: Chroma, k: int) -> float:
    """
    Compute Hit@K for the given vector store.
    """
    hits = 0
    total = 0

    for item in golden_data:
        source_text = item["text"]

        for q in item["questions"]:
            question_text = q["question"]

            results = vector_db.similarity_search(question_text, k=k)
            retrieved_texts = [r.page_content for r in results]

            if any(source_text[:200] in t for t in retrieved_texts):
                hits += 1

            total += 1

    return hits / total if total > 0 else 0.0


# =====================
# Experiment 1: K Sweep
# =====================


def test_retriever_k_sweep():
    embedder_name = EMBEDDER_NAMES[0]
    vector_db = build_vector_db(embedder_name)

    metrics = {}
    for k in K_VALUES:
        metrics[str(k)] = round(compute_hit_at_k(vector_db, k), 3)

    best_k = max(metrics, key=lambda x: metrics[x])
    if metrics[best_k] < 0.7:
        print(f"[WARN] Hit@{best_k} = {metrics[best_k]} below target")

    save_results(
        stage="retriever_k_sweep",
        metrics=metrics,
        best_key=best_k,
        setup={
            "embedding": embedder_name,
            "k_values": K_VALUES,
            "vector_store": VECTOR_STORE_NAME,
        },
    )

    save_docs(
        section_title="K Sweep",
        metrics=metrics,
        best_key=best_k,
        description="Evaluation of retriever performance across different K values "
        "using a fixed embedding model.",
    )


# =====================
# Experiment 2: Embedding Sweep
# =====================


def test_retriever_embedding_sweep():
    with open("results/retrieval_eval.json") as f:
        best_k = int(json.load(f)["best"]["key"])

    results = {}

    for embedder_name in EMBEDDER_NAMES:
        vector_db = build_vector_db(embedder_name)
        results[embedder_name] = round(compute_hit_at_k(vector_db, best_k), 3)

    best_embedder = max(results, key=lambda x: results[x])
    if results[best_embedder] < 0.7:
        print(f"[WARN] Hit@{best_k} = {results[best_embedder]} below target")

    save_results(
        stage="retriever_embedding_sweep",
        metrics=results,
        best_key=best_embedder,
        setup={
            "k": best_k,
            "embedders": EMBEDDER_NAMES,
            "vector_store": VECTOR_STORE_NAME,
        },
    )

    save_docs(
        section_title="Embedding Model Comparison",
        metrics=results,
        best_key=best_embedder,
        description=f"Comparison of embedding models at fixed K = {best_k}.",
    )


# =====================
# Experiment 3: Chunking Sweep
# =====================


def test_retriever_chunking_sweep():
    with open("results/retrieval_eval.json") as f:
        data = json.load(f)
        best_k = int(data["setup"]["k"])
        best_embedder = data["best"]["key"]

    chunking_configs = {
        "c400_o50": {"chunk_size": 400, "overlap": 50},
        "c600_o100": {"chunk_size": 600, "overlap": 100},
        "c800_o150": {"chunk_size": 800, "overlap": 150},
        "c1000_o200": {"chunk_size": 1000, "overlap": 200},
    }

    results = {}

    for name, cfg in chunking_configs.items():
        vector_db = Chroma(
            collection_name="mng_data_collection",
            embedding_function=HuggingFaceEmbeddings(
                model_name=best_embedder, model_kwargs={"device": "cpu"}
            ),
            persist_directory=(
                f"./knowledge/chromadb_{normalize_name(best_embedder)}_{name}/"
            ),
        )
        count = vector_db._collection.count()
        assert count > 0, f"Empty collection at {vector_db._persist_directory}"

        results[name] = round(compute_hit_at_k(vector_db, best_k), 3)
    best = max(results, key=lambda x: results[x])
    if results[best] < 0.7:
        print(f"[WARN] Hit@{best_k} = {results[best]} below target")

    save_results(
        stage="retriever_chunking_sweep",
        metrics=results,
        best_key=best,
        setup={
            "k": best_k,
            "embedding": best_embedder,
            "chunking": chunking_configs,
        },
    )

    save_docs(
        section_title="Chunking Strategy Comparison",
        metrics=results,
        best_key=best,
        description="Comparison of different chunk size and overlap configurations.",
    )
    save_best_retriever_config(
        k=best_k,
        embedding=best_embedder,
        chunking=chunking_configs[best],
        score=results[best],
    )


# =====================
# Persistence
# =====================


def save_results(
    stage: str,
    metrics: dict,
    best_key: str,
    setup: dict,
):
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    data = {
        "stage": stage,
        "date": datetime.utcnow().isoformat(),
        "metric": "Hit@K",
        "setup": setup,
        "results": metrics,
        "best": {
            "key": best_key,
            "score": metrics[best_key],
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_docs(
    section_title: str,
    metrics: dict,
    best_key: str,
    description: str,
):
    DOCS_PATH.parent.mkdir(exist_ok=True)

    write_header = not DOCS_PATH.exists()

    lines = []

    if write_header:
        lines += [
            "# Retriever Evaluation\n\n",
            "This document contains the results of retriever evaluation experiments.\n\n",
        ]

    lines += [
        f"## {section_title}\n\n",
        f"{description}\n\n",
        "| Parameter | Hit@K |\n",
        "|-----------|-------|\n",
    ]

    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |\n")

    lines += [
        "\n**Conclusion:**\n\n",
        f"The best configuration is **{best_key}** with **Hit@K = {metrics[best_key]}**.\n\n",
    ]

    with open(DOCS_PATH, "a", encoding="utf-8") as f:
        f.writelines(lines)


def save_best_retriever_config(
    k: int,
    embedding: str,
    chunking: dict,
    score: float,
):
    path = Path("results/best_retriever_config.json")
    path.parent.mkdir(exist_ok=True)

    data = {
        "retriever": {
            "vector_store": "chroma",
            "k": k,
            "embedding": embedding,
            "chunking": chunking,
            "hit_at_k": score,
        },
        "date": datetime.utcnow().isoformat(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
