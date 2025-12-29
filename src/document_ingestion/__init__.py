import json
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_json_documents(path: str | Path) -> List[Document]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Knowledge base file not found: {resolved_path}. "
            "Provide a JSON file with a list of objects containing a 'text' field."
        )

    with resolved_path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)

    if not isinstance(items, list):
        raise ValueError("Knowledge base JSON must be a list of objects.")

    documents: List[Document] = []
    for item in items:
        if isinstance(item, dict) and "text" in item:
            documents.append(Document(page_content=str(item["text"])))
        else:
            raise ValueError(
                "Each knowledge base entry must be an object with a 'text' field."
            )
    return documents


def split_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))
