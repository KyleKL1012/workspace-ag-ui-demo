from datetime import datetime

from travel_planner.helpers.llm_utils import invoke_llm
from travel_planner.models.available_llm_models import LLMs
from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.base_node import BaseNode
from travel_planner.prompts.prompt_handler import PromptTemplates


class LLMTripPlannerNode(BaseNode):
    def __init__(self, prompt_templates: PromptTemplates, llm_models: LLMs):
        super().__init__()
        self.prompt_templates = prompt_templates
        self.llm_models = llm_models

    async def async_run(self, state: TravelPlannerState) -> TravelPlannerState:  # type: ignore[override]
        # Try to extract travel parameters using OpenAI structured output
        tp = state.travel_params
        if (
            tp is None
            or tp.origin is None
            or tp.destination is None
            or tp.date_from is None
            or tp.date_to is None
            or tp.budget is None
        ):
            self.logger.warning("Missing travel parameters")
            raise ValueError("Missing travel parameters")

        prompt_value = self.prompt_templates.trip_planner.format_prompt(
            today=datetime.today(),
            origin=tp.origin,
            destination=str(tp.destination),
            date_from=str(tp.date_from),
            date_to=str(tp.date_to),
            budget=str(tp.budget),
        )

        # Extract travel parameters
        ai_message = await invoke_llm(
            prompt_value=prompt_value,
            response_model=None,
            llm=self.llm_models.large_model,
            messages_history=state.messages,
        )
        state.last_ai_message = str(ai_message.content)
        state.messages.append(ai_message)
        return state
