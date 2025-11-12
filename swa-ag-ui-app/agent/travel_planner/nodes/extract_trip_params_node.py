from datetime import datetime

from travel_planner.helpers.llm_utils import invoke_llm
from travel_planner.models.available_llm_models import LLMs
from travel_planner.models.state import TravelParams, TravelPlannerState
from travel_planner.nodes.base_node import BaseNode
from travel_planner.prompts.prompt_handler import PromptTemplates


class ExtractTripParamsNode(BaseNode):
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
        prompt_value = self.prompt_templates.trip_params_extraction.format_prompt(
            user_message=state.user_prompt, today=datetime.today()
        )

        # Extract travel parameters
        travel_params = await invoke_llm(
            prompt_value=prompt_value,
            response_model=TravelParams,
            llm=self.llm_models.mini_model,
            messages_history=state.messages,
        )

        if not isinstance(travel_params, TravelParams):
            self.logger.error(
                f"LLM response could not be parsed into TravelParams: {travel_params}"
            )
            raise ValueError("Invalid travel parameters")

        # Validate if we have all required information
        missing_fields = []
        for field_name, field_value in travel_params.model_dump().items():
            if field_value is None:
                missing_fields.append(field_name)

        if missing_fields:
            self.logger.info(
                f"{self.node_id} | Missing or invalid travel parameters: {missing_fields}"
            )
            # Mark state as needing user input fixing
            state.travel_params = travel_params  # Save partial params
            state.missing_trip_params = missing_fields
        else:
            # All parameters successfully extracted
            state.travel_params = travel_params
            state.missing_trip_params = []

            self.logger.info(
                "Successfully extracted travel params: %s ➜ %s | %s → %s | €%.2f",
                travel_params.origin,
                travel_params.destination,
                travel_params.date_from,
                travel_params.date_to,
                travel_params.budget,
            )
        return state
