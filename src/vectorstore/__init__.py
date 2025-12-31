from typing import Iterable

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(
    documents: Iterable[Document],
    embedding_model: str,
) -> InMemoryVectorStore:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)
    stored_documents = list(documents)
    vector_store.add_documents(stored_documents)
    vector_store._stored_documents = stored_documents
    return vector_store


def build_faiss_vector_store(
    documents: Iterable[Document],
    embedding_model: str,
    persist_dir: str,
) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    stored_documents = list(documents)
    vector_store = FAISS.from_documents(stored_documents, embeddings)
    vector_store.save_local(persist_dir)
    vector_store._stored_documents = stored_documents
    return vector_store


def load_faiss_vector_store(
    embedding_model: str,
    persist_dir: str,
) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def get_vector_store_documents(vector_store: VectorStore) -> list[Document]:
    stored_documents = getattr(vector_store, "_stored_documents", None)
    if stored_documents is not None:
        return list(stored_documents)

    docstore = getattr(vector_store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        return list(docstore._dict.values())

    return []
