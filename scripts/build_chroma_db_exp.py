from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.paths import normalize_name


PDF_PATH = "./knowledge/sources/pdf/mng_data.pdf"


EMBEDDERS = [
    "cointegrated/rubert-tiny2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

BASE_CHUNKING = {
    "chunk_size": 600,
    "overlap": 100,
}

CHUNKING_CONFIGS = {
    "c400_o50": {"chunk_size": 400, "overlap": 50},
    "c600_o100": {"chunk_size": 600, "overlap": 100},
    "c800_o150": {"chunk_size": 800, "overlap": 150},
    "c1000_o200": {"chunk_size": 1000, "overlap": 200},
}


def build_chroma_db(
    embedder_name: str,
    persist_dir: str,
    chunk_size: int,
    overlap: int,
):
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    documents = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedder_name)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="mng_data_collection",
    )
    print(f"Built DB: {persist_dir}")


if __name__ == "__main__":

    # ---------- Stage 1: base DBs (Hit@K + embedding sweep)
    for embedder in EMBEDDERS:
        build_chroma_db(
            embedder_name=embedder,
            persist_dir=f"./knowledge/chromadb_{embedder.replace('/', '_')}",
            **BASE_CHUNKING,
        )

    # ---------- Stage 2: chunking sweep DBs
    for embedder in EMBEDDERS:
        for name, cfg in CHUNKING_CONFIGS.items():
            build_chroma_db(
                embedder_name=embedder,
                persist_dir=f"./knowledge/chromadb_{normalize_name(embedder)}_{name}/",
                **cfg,
            )
