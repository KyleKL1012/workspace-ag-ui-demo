# returns a Graph object
from pathlib import Path

from dotenv import load_dotenv

from travel_planner.main import get_compiled_travel_planner_graph

load_dotenv()


def draw_graph() -> None:
    # make sure docs/ exists
    docs_path = Path(__file__).parent.parent / "docs"
    docs_path.mkdir(exist_ok=True)

    # init orchestrator
    compiled_graph = get_compiled_travel_planner_graph()

    # ⇩ note the extra `.get_graph()`
    compiled_graph.get_graph().draw_mermaid_png(output_file_path=docs_path / "langgraph.png")


if __name__ == "__main__":
    draw_graph()
    print("Graph saved to docs/langgraph.png")
