from travel_planner.helpers.llm_utils import invoke_llm
from travel_planner.models.available_llm_models import LLMs
from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.base_node import BaseNode
from travel_planner.prompts.prompt_handler import PromptTemplates


class ChitchatNode(BaseNode):
    """
    Node that handles general conversation and non-travel related queries.
    Uses LLM to generate friendly responses to user messages.
    """

    def __init__(self, prompt_templates: PromptTemplates, llm_models: LLMs):
        super().__init__()
        self.prompt_templates = prompt_templates
        self.llm_models = llm_models

    async def async_run(self, state: TravelPlannerState) -> TravelPlannerState:  # type: ignore[override]
        # Generate chitchat response using LLM
        prompt_value = self.prompt_templates.chitchat_response.format_prompt(
            user_message=state.user_prompt
        )

        # Get response from LLM as BaseMessage (no structured output needed)
        ai_message = await invoke_llm(
            prompt_value=prompt_value,
            response_model=None,
            llm=self.llm_models.mini_model,
            messages_history=state.messages,
        )
        if ai_message is None or not hasattr(ai_message, "content"):
            self.logger.error("LLM response is invalid or missing content")
            raise ValueError("Invalid LLM response for chitchat")

        # Store the chitchat response in state
        state.last_ai_message = str(ai_message.content)

        state.messages.append(ai_message)
        self.logger.info(f"{self.node_id} | Generated chitchat response")

        return state
