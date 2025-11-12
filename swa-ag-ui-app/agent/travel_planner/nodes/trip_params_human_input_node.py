from langchain_core.messages import HumanMessage

from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.base_node import BaseNode


class TripParamsHumanInputNode(BaseNode):
    """
    Human-in-the-loop node that asks the user for missing travel parameters.
    This node will interrupt the execution and wait for human input.
    """

    def __init__(self) -> None:
        super().__init__()

    async def async_run(self, state: TravelPlannerState) -> TravelPlannerState:
        """
        This node append the extra input coming from user
        """
        extra_input_from_user = f"Additional Input: {state.user_prompt}"
        human_message = HumanMessage(content=extra_input_from_user)
        state.messages.append(human_message)
        return state
