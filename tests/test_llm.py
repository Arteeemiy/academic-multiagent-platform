import json
from pathlib import Path
from datetime import datetime

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
import getpass
from dotenv import load_dotenv

load_dotenv()

# =====================
# Config
# =====================

LLM_MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
    "mistral-large-2512",
]

JUDGE_MODEL = "mistral-large-2512"

DATA_PATH = Path("tests/golden_contexts.json")
RESULTS_PATH = Path("results/llm_eval.json")
DOCS_PATH = Path("docs/llm_eval.md")


# =====================
# Prompts
# =====================

ANSWER_PROMPT = """
Answer the question strictly using the provided context.
If the context is insufficient, say "INSUFFICIENT CONTEXT".
"""

JUDGE_PROMPT = """
You are an expert evaluator.

Evaluate the ANSWER using the CONTEXT and QUESTION.

Score from 0 to 1:
- 1.0 = fully correct, grounded, no hallucinations
- 0.5 = partially correct or incomplete
- 0.0 = incorrect or hallucinated

Output ONLY valid JSON:
{"score": float}
"""


def clean_json(raw: str) -> str:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]

    if raw.startswith("json"):
        raw = raw[4:]

    return raw.strip()


# =====================
# Core logic
# =====================


def run_llm(llm_name: str, data: list) -> float:
    llm = ChatMistralAI(model=llm_name, temperature=0, max_retries=7)
    judge = ChatMistralAI(model=JUDGE_MODEL, temperature=0, max_retries=7)

    scores = []

    for item in data:
        question = item["question"]
        context = item["golden_context"]["text"]

        answer = llm.invoke(
            [
                SystemMessage(content=ANSWER_PROMPT),
                HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
            ]
        ).content

        judgment = judge.invoke(
            [
                SystemMessage(content=JUDGE_PROMPT),
                HumanMessage(
                    content=f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
                ),
            ]
        ).content

        judgment = clean_json(judgment)
        score = json.loads(judgment)["score"]
        scores.append(score)

    return round(sum(scores) / len(scores), 3)


# =====================
# Test
# =====================


def test_llm_model_sweep():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    metrics = {}

    for model in LLM_MODELS:
        metrics[model] = run_llm(model, data)

    best_model = max(metrics, key=lambda x: metrics[x])

    save_results(
        stage="llm_model_sweep",
        metrics=metrics,
        best_key=best_model,
        setup={
            "judge": JUDGE_MODEL,
            "oracle_context": True,
        },
    )

    save_docs(
        section_title="LLM Model Comparison (Oracle Context)",
        metrics=metrics,
        best_key=best_model,
        description="Evaluation of LLM answer quality using golden (oracle) context.",
    )

    save_best_llm_config(best_model, metrics[best_model])


# =====================
# Persistence
# =====================


def save_results(stage, metrics, best_key, setup):
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": stage,
                "date": datetime.utcnow().isoformat(),
                "metric": "LLMScore",
                "setup": setup,
                "results": metrics,
                "best": {
                    "model": best_key,
                    "score": metrics[best_key],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_docs(section_title, metrics, best_key, description):
    DOCS_PATH.parent.mkdir(exist_ok=True)
    write_header = not DOCS_PATH.exists()

    lines = []

    if write_header:
        lines += [
            "# LLM Evaluation\n\n",
            "This document contains LLM evaluation results.\n\n",
        ]

    lines += [
        f"## {section_title}\n\n",
        f"{description}\n\n",
        "| Model | Score |\n",
        "|-------|-------|\n",
    ]

    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |\n")

    lines += [
        "\n**Conclusion:**\n\n",
        f"Best model: **{best_key}** with score **{metrics[best_key]}**.\n\n",
    ]

    with open(DOCS_PATH, "a", encoding="utf-8") as f:
        f.writelines(lines)


def save_best_llm_config(model, score):
    path = Path("results/best_llm_config.json")
    path.parent.mkdir(exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "llm": {
                    "model": model,
                    "score": score,
                },
                "date": datetime.utcnow().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
