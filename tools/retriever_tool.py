import sys

print("Retriever Python:", sys.executable)

from typing import Type, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# ===== Input schema =====
class RetrieverInput(BaseModel):
    query: str = Field(
        ..., description="Search query for retrieving relevant document chunks."
    )
    k: int = Field(9, description="Number of chunks to retrieve.")


# ===== Tool =====
class ChromaRetrieverTool(BaseTool):
    name: str = "document_retriever"
    description: str = (
        "Retrieves relevant document chunks from the coursework knowledge base. "
        "Use this tool to gather factual context before writing sections of the coursework."
    )
    args_schema: Type[BaseModel] = RetrieverInput

    def _run(self, query: str, k: int = 9) -> str:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE",
            model_kwargs={"device": "cpu"},
        )

        db = Chroma(
            collection_name="mng_data_collection",
            embedding_function=embeddings,
            persist_directory="./knowledge/chromadb_sentence-transformers_LaBSE_c1000_o200/",
        )

        docs = db.similarity_search(query, k=k)

        if not docs:
            return "Недостаточно данных в источниках."

        # 🔴 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
        # Возвращаем ТОЛЬКО чистый текст
        return "\n\n".join(doc.page_content.strip() for doc in docs)
