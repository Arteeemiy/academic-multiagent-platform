import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
import getpass
import os


if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")


PDF_PATH = "./knowledge/sources/pdf/mng_data.pdf"
OUTPUT_PATH = "./tests/golden_questions.json"


llm = ChatMistralAI(
    model="mistral-large-2512",
    temperature=0,
    max_retries=2,
)


SYSTEM_PROMPT = """
You are an expert academic assistant.

Your task is to generate 1–2 meaningful questions
that can be answered strictly using the provided document fragment.

Rules:
- Questions must be specific and factual
- Questions must be useful for testing a semantic search system
- Do NOT ask questions that require external knowledge
- Do NOT include answers
- Output ONLY valid JSON
- Output format:
[
  {"question_id": "q1", "question": "..."},
  {"question_id": "q2", "question": "..."}
]
"""
FIX_PROMPT = """
The following JSON is invalid or does not match the required schema.

Required schema:
[
  {"question_id": "q1", "question": "string"},
  {"question_id": "q2", "question": "string"}
]

Fix the JSON.
Return ONLY valid JSON.
Do not add explanations.
"""


def validate_questions(obj):
    if not isinstance(obj, list):
        return False

    for item in obj:
        if not isinstance(item, dict):
            return False
        if "question_id" not in item or "question" not in item:
            return False
        if not isinstance(item["question"], str):
            return False

    return True


def clean_json(raw: str) -> str:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]

    if raw.startswith("json"):
        raw = raw[4:]

    return raw.strip()


def fix_json(llm, raw_json: str) -> str:
    messages = [
        SystemMessage(content=FIX_PROMPT),
        HumanMessage(content=raw_json),
    ]
    return llm.invoke(messages).content


def generate_questions(text: str, retries: int = 2):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"DOCUMENT:\n{text}\n\nGenerate questions:"),
    ]

    raw = llm.invoke(messages).content
    raw = clean_json(raw)

    for _ in range(retries):
        try:
            data = json.loads(raw)
            if validate_questions(data):
                return data
        except json.JSONDecodeError:
            pass

        raw = fix_json(llm, raw)
        raw = clean_json(raw)

    raise ValueError(f"Unfixable JSON from LLM:\n{raw}")


def main():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    dataset = []

    for i, doc in enumerate(documents):
        questions = generate_questions(doc.page_content)

        dataset.append(
            {
                "chunk_id": i,
                "text": doc.page_content,
                "questions": questions,
            }
        )

        print(f"[OK] Chunk {i}: {questions}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
