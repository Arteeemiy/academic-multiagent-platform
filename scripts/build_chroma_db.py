from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import chromadb
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def chroma_db_create(collection_name: str) -> Chroma:
    loader = PyPDFLoader("./knowledge/sources/pdf/mng_data.pdf")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")

    client = chromadb.PersistentClient(
        path="./knowledge/chromadb_cointegrated_rubert-tiny2/"
    )
    collection = client.get_or_create_collection("mng_data_collection")
    vector_store_from_client = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    vector_store_from_client.add_documents(docs)

    return vector_store_from_client


if __name__ == "__main__":
    chroma_db_create(collection_name="mng_data_collection")
    print("Chroma DB created successfully.")
