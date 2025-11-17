import logging
import json
from typing import Any, List, Optional
from typing_extensions import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AgentState(MessagesState):
    proverbs: List[str] = []
    tools: List[Any] = []

@tool
def get_weather(location: str):
    """
    Fetch the weather for a specific location.

    Parameters:
        location (str): Target location name.

    Returns:
        str: Weather information.
    """
    return f"🌤 The weather for {location} is 70°F, clear skies."

backend_tools = [get_weather]
backend_tool_names = [tool.name for tool in backend_tools]

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "approval_node", "__end__"]]:
    """Main chat node that interacts with the LLM"""
    model = ChatGroq(model="openai/gpt-oss-120b")
    tools_to_bind = [t for t in state.get("tools", []) if t is not None] + backend_tools
    model_with_tools = model.bind_tools(tools_to_bind, parallel_tool_calls=False)

    system_message = SystemMessage(
        content=f"You are a helpful assistant. The current proverbs are {state.get('proverbs', [])}."
    )

    messages = state.get("messages", [])
    response = await model_with_tools.ainvoke([system_message, *messages], config)

    # Check if approval is needed for specific tools
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.get("name") == "get_weather":
                # Route to approval node for weather requests
                return Command(goto="approval_node", update={"messages": [response]})
    
    # Route to tool node for other tools
    if route_to_tool_node(response):
        return Command(goto="tool_node", update={"messages": [response]})

    return Command(goto=END, update={"messages": [response]})


async def approval_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """Handle user approval for sensitive operations"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if not last_message:
        return Command(goto=END)
    
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return Command(goto=END)
    
    # Find the weather tool call
    weather_call = next((tc for tc in tool_calls if tc.get("name") == "get_weather"), None)
    if not weather_call:
        return Command(goto=END)
    
    location = weather_call.get("args", {}).get("location", "unknown")
    
    # Interrupt and wait for user approval
    interrupt_payload = {
        "action": "confirm_weather_request",
        "message": f"Do you approve fetching weather for {location}?",
        "location": location
    }
    
    logger.info("\nHITL Sending to frontend:")
    logger.info(json.dumps(interrupt_payload, indent=2))
    
    approval = interrupt(interrupt_payload)

    logger.info(f"\nHITL Received from frontend: '{approval}'\n")
    
    if approval == "approved":
        # User approved, proceed to tool execution
        return Command(goto="tool_node")
    else:
        # User rejected, return cancellation message
        cancel_msg = AIMessage(content=f"x Weather request for {location} was cancelled by user.")
        return Command(goto=END, update={"messages": [cancel_msg]})


def route_to_tool_node(response: BaseMessage) -> bool:
    """Check if response contains tool calls that should be executed"""
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False


# Build the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("approval_node", approval_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

# Compile the graph
graph = workflow.compile()