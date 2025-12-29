from typing import Callable

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.language_models import BaseLanguageModel

from src.state import State


def classify(state: State) -> dict:
    _ = "advanced" in state["question"].lower()
    return {"question": state["question"]}


def make_retrieve_node(
    vector_store: InMemoryVectorStore, k: int
) -> Callable[[State], dict]:
    def retrieve(state: State) -> dict:
        retrieved_docs = vector_store.similarity_search(state["question"], k=k)
        return {"context": retrieved_docs}

    return retrieve


def make_generate_node(
    llm: BaseLanguageModel, prompt_template: str
) -> Callable[[State], dict]:
    def generate(state: State) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt_filled = prompt_template.format(
            question=state["question"], context=docs_content
        )
        response = llm.invoke([{"role": "user", "content": prompt_filled}])
        return {"answer": response.content}

    return generate


def refine(state: State) -> dict:
    refined_answer = f"{state['answer']}\n\n[Refined for clarity and completeness]"
    return {"answer": refined_answer}
