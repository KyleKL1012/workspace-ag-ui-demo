"""
This is the main entry point for the travel planner agent.
It integrates the travel planner graph with ag-ui.
"""

from dotenv import load_dotenv

from travel_planner.graphs.travel_planner_graph import TravelPlannerGraph
from travel_planner.helpers.llm_utils import get_available_llms
from travel_planner.nodes.node_factory import NodeFactory
from travel_planner.prompts.prompt_handler import PromptTemplates
from travel_planner.settings.settings_handler import AppSettings

# Load environment variables
load_dotenv()

# Initialize the travel planner components
prompt_templates = PromptTemplates.read_from_yaml()
settings = AppSettings.read_from_yaml()
llm_models = get_available_llms(settings=settings.groq)
node_factory = NodeFactory(prompt_templates=prompt_templates, llm_models=llm_models)

# Build the graph
travel_planner_graph = TravelPlannerGraph(node_factory=node_factory)
built_graph = travel_planner_graph.build_graph()

# Compile the graph with human-in-the-loop support
# Note: LangGraph API handles persistence automatically, no custom checkpointer needed
graph = built_graph.compile(
    interrupt_before=[
        node_factory.trip_params_human_input_node.node_id  # Human in the loop
    ],
)
