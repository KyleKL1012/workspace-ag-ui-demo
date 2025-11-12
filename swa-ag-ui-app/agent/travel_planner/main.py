from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
# from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from travel_planner.graphs.travel_planner_graph import TravelPlannerGraph
from travel_planner.helpers.llm_utils import get_available_llms
from travel_planner.nodes.node_factory import NodeFactory
from travel_planner.prompts.prompt_handler import PromptTemplates
from travel_planner.settings.settings_handler import AppSettings


def get_compiled_travel_planner_graph() -> CompiledStateGraph:
    load_dotenv()
    prompt_templates = PromptTemplates.read_from_yaml()
    settings = AppSettings.read_from_yaml()
    llm_models = get_available_llms(settings=settings.groq)
    node_factory = NodeFactory(prompt_templates=prompt_templates, llm_models=llm_models)
    graph = TravelPlannerGraph(node_factory=node_factory)
    built_graph = graph.build_graph()
    compiled_graph = built_graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=[
            node_factory.trip_params_human_input_node.node_id  # Human in the loop
        ],
    )
    return compiled_graph


def get_config(thread_id: str) -> RunnableConfig:
    # langfuse_handler = CallbackHandler()
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        # callbacks=[langfuse_handler],
        # metadata={"langfuse_session_id": thread_id},
    )
    return config
