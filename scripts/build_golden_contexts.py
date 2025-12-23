import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
import getpass
import os

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")


PDF_PATH = "./knowledge/sources/pdf/mng_data.pdf"
QUESTIONS_PATH = "./tests/golden_questions.json"
OUTPUT_PATH = Path("./tests/golden_contexts.json")

CHUNK_SIZE = 1000
OVERLAP = 200

llm = ChatMistralAI(
    model="mistral-large-2512",
    temperature=0,
    max_retries=2,
)

SYSTEM_PROMPT = """
You are an expert evaluator.

You will be given:
- a QUESTION
- a list of DOCUMENT CHUNKS with chunk_id

Task:
Select the ONE chunk_id that contains sufficient information to answer the question.
If none contain the answer, return "NONE".

Output ONLY valid JSON:
{"chunk_id": <int or "NONE">}
"""
FIX_PROMPT = """
The JSON below is invalid.

Required format:
{"chunk_id": <int or "NONE">}

Fix the JSON.
Return ONLY valid JSON.
"""


def clean_json(raw: str) -> str:
    if not raw:
        return ""

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]

    if raw.startswith("json"):
        raw = raw[4:]

    return raw.strip()


def validate_chunk_selection(obj):
    if not isinstance(obj, dict):
        return False
    if "chunk_id" not in obj:
        return False
    if obj["chunk_id"] == "NONE":
        return True
    return isinstance(obj["chunk_id"], int)


def main():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    dataset = []

    for item in questions:
        source_text = item["text"]

        for q in item["questions"]:
            question = q["question"]

            chunk_payload = [
                {
                    "chunk_id": c.metadata["chunk_id"],
                    "text": c.page_content,
                }
                for c in chunks
            ]

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": question,
                            "chunks": chunk_payload,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]

            raw = llm.invoke(messages).content
            raw = clean_json(raw)

            for _ in range(2):
                try:
                    data = json.loads(raw)
                    if validate_chunk_selection(data):
                        break
                except json.JSONDecodeError:
                    pass

                raw = llm.invoke(
                    [
                        SystemMessage(content=FIX_PROMPT),
                        HumanMessage(content=raw),
                    ]
                ).content
                raw = clean_json(raw)
            else:
                raise ValueError(f"Unfixable LLM output:\n{raw}")

            if data["chunk_id"] == "NONE":
                continue

            cid = int(data["chunk_id"])
            chunk = chunks[cid]

            dataset.append(
                {
                    "question": question,
                    "golden_context": {
                        "chunk_id": cid,
                        "text": chunk.page_content,
                    },
                }
            )

            print(f"[OK] Q → chunk {cid}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved golden contexts → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
