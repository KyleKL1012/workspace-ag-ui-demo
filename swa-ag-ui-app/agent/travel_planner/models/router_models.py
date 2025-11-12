from enum import Enum

from pydantic import BaseModel, Field


class Routes(Enum):
    TRAVEL_PLANNER = "travel_planner"
    CHITCHAT = "chitchat"
    ESCALATION = "escalation"
    TURKISH_AIRLINES = "turkish_airlines"


class RoutingDecision(BaseModel):
    """Model for the router node to decide the task type"""

    predicted_route: Routes = Field(description="The type of task based on user input")
    reasoning: str = Field(description="Brief explanation of why this route was chosen")
