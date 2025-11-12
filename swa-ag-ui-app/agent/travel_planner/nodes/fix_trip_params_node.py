from travel_planner.helpers.llm_utils import invoke_llm
from travel_planner.models.available_llm_models import LLMs
from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.base_node import BaseNode
from travel_planner.prompts.prompt_handler import PromptTemplates


class FixTripParamsNode(BaseNode):
    """
    First node in the graph: receives **raw** user message and attempts to extract
    travel parameters using OpenAI structured output. If parsing fails or is incomplete,
    the state is marked for user input fixing.
    """

    def __init__(self, prompt_templates: PromptTemplates, llm_models: LLMs):
        super().__init__()
        self.prompt_templates = prompt_templates
        self.llm_models = llm_models

    async def async_run(self, state: TravelPlannerState) -> TravelPlannerState:  # type: ignore[override]
        # Try to extract travel parameters using OpenAI structured output
        prompt_value = self.prompt_templates.fix_trip_params_extraction.format_prompt(
            missing_trip_params=state.missing_trip_params
        )
        user_message = prompt_value.to_messages()[1]
        ai_message = await invoke_llm(
            prompt_value=prompt_value,
            llm=self.llm_models.mini_model,
            messages_history=state.messages,
        )

        # Add user and ai message to the history
        state.messages.append(user_message)
        state.messages.append(ai_message)
        state.last_ai_message = str(ai_message.content)
        return state
