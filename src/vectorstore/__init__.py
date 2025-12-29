from typing import Iterable

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


def build_vector_store(
    documents: Iterable[Document],
    embedding_model: str,
) -> InMemoryVectorStore:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(list(documents))
    return vector_store
