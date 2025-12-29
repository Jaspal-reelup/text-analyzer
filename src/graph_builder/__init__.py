from langgraph.graph import START, StateGraph

from src.nodes import classify, make_generate_node, make_retrieve_node, refine
from src.state import State


def build_graph(vector_store, llm, prompt_template, k: int):
    graph_builder = StateGraph(State)
    graph_builder.add_node("classify", classify)
    graph_builder.add_node("retrieve", make_retrieve_node(vector_store, k))
    graph_builder.add_node("generate", make_generate_node(llm, prompt_template))
    graph_builder.add_node("refine", refine)

    graph_builder.add_edge(START, "classify")
    graph_builder.add_edge("classify", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "refine")

    return graph_builder.compile(), graph_builder


def visualize_langgraph_clean(graph_builder):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Visualization requires networkx and matplotlib."
        ) from exc

    G = nx.DiGraph()
    for node_name in graph_builder.nodes:
        G.add_node(node_name)
    for src, tgt in graph_builder.edges:
        G.add_edge(src, tgt)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=1.2)

    node_styles = {
        "__start__": {"color": "#666666", "size": 3500},
        "classify": {"color": "#56c2ff", "size": 3300},
        "retrieve": {"color": "#75ff90", "size": 3300},
        "generate": {"color": "#ff8888", "size": 3300},
        "refine": {"color": "#b996fa", "size": 3500},
    }
    node_colors = [
        node_styles.get(node, {"color": "#cccccc"})["color"] for node in G.nodes()
    ]
    node_sizes = [
        node_styles.get(node, {"size": 2700})["size"] for node in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#303030",
        alpha=0.93,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-",
        arrowsize=25,
        width=3,
        edge_color="#555",
        alpha=0.75,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(
        G, pos, font_size=17, font_weight="bold", font_family="sans-serif"
    )

    plt.title("LangGraph Workflow", fontsize=18, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
