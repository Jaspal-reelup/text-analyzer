import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import (
    CUSTOM_PROMPT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KNOWLEDGE_BASE_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_PDF_DIR,
    DEFAULT_VECTORSTORE_DIR,
)
from src.document_ingestion import load_json_documents, load_pdf_documents, split_documents
from src.graph_builder import build_graph, visualize_langgraph_clean
from src.vectorstore import (
    build_faiss_vector_store,
    build_vector_store,
    get_vector_store_documents,
    load_faiss_vector_store,
)


def init_llm():
    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        init_chat_model = None

    if init_chat_model:
        return init_chat_model(f"openai:{DEFAULT_LLM_MODEL}", temperature=DEFAULT_TEMPERATURE)

    return ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=DEFAULT_TEMPERATURE)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="LangGraph RAG demo")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render the LangGraph workflow (requires networkx/matplotlib).",
    )
    parser.add_argument(
        "--use-pdfs",
        action="store_true",
        help="Load and index PDFs from a directory (for large corpora).",
    )
    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_VECTORSTORE_DIR,
        help="Directory to save/load the FAISS index.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the FAISS index from PDFs.",
    )
    parser.add_argument(
        "--print-vectorstore",
        action="store_true",
        help="Print stored document contents from the vector store and exit.",
    )
    args = parser.parse_args()

    if args.use_pdfs:
        persist_dir = Path(args.persist_dir)
        if persist_dir.exists() and not args.rebuild_index:
            vector_store = load_faiss_vector_store(
                DEFAULT_EMBEDDING_MODEL,
                str(persist_dir),
            )
        else:
            documents = load_pdf_documents(args.pdf_dir)
            splits = split_documents(documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            persist_dir.mkdir(parents=True, exist_ok=True)
            vector_store = build_faiss_vector_store(
                splits,
                DEFAULT_EMBEDDING_MODEL,
                str(persist_dir),
            )
    else:
        knowledge_path = os.getenv("KNOWLEDGE_BASE_PATH", DEFAULT_KNOWLEDGE_BASE_PATH)
        documents = load_json_documents(knowledge_path)
        splits = split_documents(documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        vector_store = build_vector_store(splits, DEFAULT_EMBEDDING_MODEL)
    llm = init_llm()

    graph, graph_builder = build_graph(vector_store, llm, CUSTOM_PROMPT, DEFAULT_RETRIEVAL_K)

    if args.print_vectorstore:
        stored_docs = get_vector_store_documents(vector_store)
        for idx, doc in enumerate(stored_docs, start=1):
            print(f"[{idx}] {doc.page_content}")
        return

    if args.visualize:
        visualize_langgraph_clean(graph_builder)

    print("RAG system is ready. Type 'exit' to quit.")
    while True:
        question = input("Enter your question: ").strip()
        if question.lower() in ("exit", "quit", "stop"):
            print("Exiting program. Goodbye!")
            break
        if not question:
            continue

        response = graph.invoke({"question": question})
        answer = response.get("answer", "No answer generated.")
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
