import argparse
import os

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
)
from src.document_ingestion import load_json_documents, split_documents
from src.graph_builder import build_graph, visualize_langgraph_clean
from src.vectorstore import build_vector_store


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
    args = parser.parse_args()

    knowledge_path = os.getenv("KNOWLEDGE_BASE_PATH", DEFAULT_KNOWLEDGE_BASE_PATH)
    documents = load_json_documents(knowledge_path)
    splits = split_documents(documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    vector_store = build_vector_store(splits, DEFAULT_EMBEDDING_MODEL)
    llm = init_llm()

    graph, graph_builder = build_graph(vector_store, llm, CUSTOM_PROMPT, DEFAULT_RETRIEVAL_K)

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
