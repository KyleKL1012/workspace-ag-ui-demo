from datetime import date
from typing import Annotated, Any

from langgraph.graph.message import BaseMessage, add_messages
from pydantic import BaseModel, Field

from travel_planner.models.router_models import RoutingDecision


class HotelInfo(BaseModel):
    name: str
    address: str
    price_per_nfght: float
    currency: str = "EUR"
    rating: float | None = None
    provider_id: str


class HotelAPIParams(BaseModel):
    cityCode: str = Field(..., description="IATA city code (e.g. PAR)")
    checkInDate: date
    checkOutDate: date
    adults: int
    roomQuantity: int = 1
    currency: str = "EUR"
    priceRange: str = Field(..., description="Format 'min-max', e.g. '0-150'")
    sort: str = "PRICE"


class TravelParams(BaseModel):
    origin: str | None = None
    destination: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    budget: float | None = None


class TravelPlannerState(BaseModel):
    """State object produced by ExtractTripParamsNode."""

    user_prompt: str = ""  # Will be extracted from messages if empty
    messages: Annotated[list[BaseMessage], add_messages] = []
    travel_params: TravelParams | None = None
    missing_trip_params: list[str] = []
    routing_decision: RoutingDecision | None = None
    last_ai_message: str | None = None
    # Tools from ag-ui (optional)
    tools: list[Any] = []
