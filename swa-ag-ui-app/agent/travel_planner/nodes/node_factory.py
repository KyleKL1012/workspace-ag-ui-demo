from functools import cached_property

from travel_planner.models.available_llm_models import LLMs
from travel_planner.nodes.chitchat_node import ChitchatNode
from travel_planner.nodes.escalation_node import EscalationNode
from travel_planner.nodes.extract_trip_params_node import ExtractTripParamsNode
from travel_planner.nodes.fix_trip_params_node import FixTripParamsNode
from travel_planner.nodes.llm_trip_planner_node import LLMTripPlannerNode
from travel_planner.nodes.router_node import RouterNode
from travel_planner.nodes.trip_params_human_input_node import TripParamsHumanInputNode
from travel_planner.nodes.turkish_airlines_node import TurkishAirlinesNode
from travel_planner.prompts.prompt_handler import PromptTemplates


class NodeFactory:
    def __init__(self, prompt_templates: PromptTemplates, llm_models: LLMs):
        # Common variables for some nodes
        self.prompt_templates = prompt_templates
        self.llm_models = llm_models

    @cached_property
    def extract_trip_params_node(self) -> ExtractTripParamsNode:
        return ExtractTripParamsNode(
            prompt_templates=self.prompt_templates, llm_models=self.llm_models
        )

    @cached_property
    def fix_trip_params_node(self) -> FixTripParamsNode:
        return FixTripParamsNode(prompt_templates=self.prompt_templates, llm_models=self.llm_models)

    @cached_property
    def trip_params_human_input_node(self) -> TripParamsHumanInputNode:
        return TripParamsHumanInputNode()

    @cached_property
    def router_node(self) -> RouterNode:
        return RouterNode(prompt_templates=self.prompt_templates, llm_models=self.llm_models)

    @cached_property
    def chitchat_node(self) -> ChitchatNode:
        return ChitchatNode(prompt_templates=self.prompt_templates, llm_models=self.llm_models)

    @cached_property
    def escalation_node(self) -> EscalationNode:
        return EscalationNode()

    @cached_property
    def turkish_airlines_node(self) -> TurkishAirlinesNode:
        return TurkishAirlinesNode(
            prompt_templates=self.prompt_templates, llm_models=self.llm_models
        )

    @cached_property
    def llm_trip_planner_node(self) -> LLMTripPlannerNode:
        return LLMTripPlannerNode(
            prompt_templates=self.prompt_templates, llm_models=self.llm_models
        )
