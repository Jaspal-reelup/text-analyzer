import json
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


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


def load_pdf_documents(pdf_dir: str | Path) -> List[Document]:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_path}. Create it and add PDF files."
        )

    pdf_files = sorted(pdf_path.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under: {pdf_path}")

    documents: List[Document] = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
    return documents
