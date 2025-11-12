from typing import Generic, TypeVar

from pydantic import BaseModel

from travel_planner.helpers.logs import get_logger

State = TypeVar("State", bound=BaseModel)


class BaseNode(Generic[State]):
    def __init__(self) -> None:
        self.logger = get_logger()

    @property
    def node_id(self) -> str:
        return self.__class__.__name__

    # Either implement async_run or run or both.
    # That's why it's not annotated with @abstractmethod.
    async def async_run(self, state: State) -> State:
        return state

    def run(self, state: State) -> State:
        return state
