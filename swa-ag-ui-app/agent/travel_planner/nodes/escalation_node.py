from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.base_node import BaseNode


class EscalationNode(BaseNode):
    """
    Node that handles escalation requests when users want to contact a human agent.
    Returns a placeholder message indicating the request has been escalated.
    """

    def __init__(self) -> None:
        super().__init__()

    async def async_run(self, state: TravelPlannerState) -> TravelPlannerState:  # type: ignore[override]
        # Set escalation response
        escalation_message = "[..Contacting with Agent]"
        state.last_ai_message = escalation_message

        self.logger.info(f"{self.node_id} | User request escalated to human agent")

        return state
