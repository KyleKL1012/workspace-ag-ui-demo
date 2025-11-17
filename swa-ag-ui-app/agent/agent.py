from typing import Any, List, Optional
from typing_extensions import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

class AgentState(MessagesState):
    proverbs: List[str] = []
    tools: List[Any] = []

@tool
def get_weather(location: str, approved: Optional[bool] = None):
    """
    Fetch the weather for a specific location.

    Parameters:
        location (str): Target location name.
        approved (Optional[bool]): Internal approval flag used for interrupt flow.

    Returns:
        str | None: Weather info or interrupt request.
    """
    if approved is None:
        user_decision = interrupt({
            "action": "confirm_weather_request",
            "message": f"Do you approve fetching weather for {location}?"
        })

        if isinstance(user_decision, str):
            approved = user_decision
        else:
            approved = "rejected"

        if approved != "approved":
            return f":x: Weather request for {location} cancelled by user."

        return None

    return f"🌤 The weather for {location} is 70°F, clear skies."

backend_tools = [get_weather]
backend_tool_names = [tool.name for tool in backend_tools]

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    model = ChatGroq(model="openai/gpt-oss-120b")
    tools_to_bind = [t for t in state.get("tools", []) if t is not None] + backend_tools
    model_with_tools = model.bind_tools(tools_to_bind, parallel_tool_calls=False)

    system_message = SystemMessage(
        content=f"You are a helpful assistant. The current proverbs are {state.get('proverbs', [])}."
    )

    messages = state.get("messages", [])
    response = await model_with_tools.ainvoke([system_message, *messages], config)

    if route_to_tool_node(response):
        return Command(goto="tool_node", update={"messages": [response]})

    return Command(goto=END, update={"messages": [response]})


def route_to_tool_node(response: BaseMessage) -> bool:
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False

# workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

# compile without custom checkpointer, uses LangGraph built-in persistence
graph = workflow.compile()

# ----------------------
# Example for resume (local/cloud agnostic):
# from langgraph_api.client import GraphClient
# client = GraphClient()
#
# run once:
# result = await client.run_graph('sample_agent', input={'messages': []})
# state_id = result.state_id
#
# resume later:
# resume_result = await client.resume_graph('sample_agent', state_id=state_id, input={'messages': []})