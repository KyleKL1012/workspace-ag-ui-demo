import logging
import json
import asyncio
from typing import Any, List, Optional
from typing_extensions import Literal
from uuid import uuid4
import httpx

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphInterrupt

# A2A Client imports
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# HR Agent A2A Server URL
HR_AGENT_URL = "http://localhost:10000"

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

@tool
def delegate_to_hr_agent(query: str):
    """
    Delegate HR onboarding related questions to the specialized HR Onboarding Agent.

    The HR Agent runs as a separate A2A server and can help with new employee onboarding,
    answering questions about company policies, setup procedures, first week schedules,
    and other HR-related inquiries.

    Parameters:
        query (str): The HR onboarding question or request from the user.

    Returns:
        str: Confirmation that the request will be delegated.
    """
    return f"Delegating to HR Agent: {query}"

backend_tools = [get_weather, delegate_to_hr_agent]
backend_tool_names = [tool.name for tool in backend_tools]

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "approval_node", "hr_agent_node", "__end__"]]:
    """Main chat node that interacts with the LLM"""
    model = ChatGroq(model="openai/gpt-oss-120b")
    tools_to_bind = [t for t in state.get("tools", []) if t is not None] + backend_tools
    model_with_tools = model.bind_tools(tools_to_bind, parallel_tool_calls=False)

    system_message = SystemMessage(
        content=f"You are a helpful orchestrator assistant that can delegate tasks to specialized agents. "
                f"When users ask HR onboarding questions, use the delegate_to_hr_agent tool. "
                f"The current proverbs are {state.get('proverbs', [])}."
    )

    messages = state.get("messages", [])
    response = await model_with_tools.ainvoke([system_message, *messages], config)

    # Check for tool calls and route appropriately
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")

            # Delegate to HR agent subgraph
            if tool_name == "delegate_to_hr_agent":
                logger.info(f"\n[ORCHESTRATOR] Delegating to HR Agent subgraph...")
                query = tool_call.get("args", {}).get("query", "")
                # Route to HR agent subgraph - interrupts will bubble up automatically
                return Command(goto="hr_agent_node", update={"messages": [response]})

            # Require approval for weather requests
            elif tool_name == "get_weather":
                return Command(goto="approval_node", update={"messages": [response]})

    # Route to tool node for other tools
    if route_to_tool_node(response):
        return Command(goto="tool_node", update={"messages": [response]})

    return Command(goto=END, update={"messages": [response]})


async def hr_agent_node(state: AgentState, config: RunnableConfig) -> Command[Literal["chat_node", "__end__"]]:
    """
    HR Agent Node (via A2A Client)

    This node calls the HR Onboarding Agent using the proper A2A client.
    When the HR agent needs approval (HITL), the interrupt bubbles up to the orchestrator level.
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message:
        return Command(goto=END)

    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return Command(goto=END)

    # Find the delegate_to_hr_agent tool call
    hr_call = next((tc for tc in tool_calls if tc.get("name") == "delegate_to_hr_agent"), None)
    if not hr_call:
        return Command(goto=END)

    query = hr_call.get("args", {}).get("query", "")
    logger.info(f"\n[ORCHESTRATOR] Calling HR Agent A2A server with query: {query}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=20)) as httpx_client:
        try:
            # Initialize A2A Card Resolver and fetch agent card
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=HR_AGENT_URL
            )

            logger.info(f"[ORCHESTRATOR] Fetching agent card from {HR_AGENT_URL}")
            agent_card = await resolver.get_agent_card()
            logger.info(f"[ORCHESTRATOR] Successfully fetched agent card: {agent_card.name}")

            # Initialize A2A Client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

            # Send first message to HR agent
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': query}],
                    'message_id': uuid4().hex,
                },
            }

            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )

            logger.info(f"[ORCHESTRATOR] Sending message to HR agent")
            response = await client.send_message(request)

            # Extract task info from response
            task_id = response.root.result.id
            context_id = response.root.result.context_id
            task_status_obj = response.root.result.status

            # The status object has a 'state' attribute with the actual TaskState enum
            task_state = task_status_obj.state if hasattr(task_status_obj, 'state') else None

            logger.info(f"[ORCHESTRATOR] HR agent task created: {task_id}, state: {task_state}")

            # Check if HR agent requires approval (input_required state)
            # TaskState enum value is 'input-required' (with hyphen)
            if task_state and (task_state.value == "input-required" or str(task_state) == "TaskState.input_required"):
                logger.info("\n[ORCHESTRATOR] HR Agent requires user input - triggering interrupt!")

                # Extract the message from HR agent
                # The message is in status.message.parts, not result.output.parts
                hr_message = "HR Agent requires approval"
                if hasattr(task_status_obj, 'message') and task_status_obj.message and hasattr(task_status_obj.message, 'parts'):
                    for part in task_status_obj.message.parts:
                        if hasattr(part, 'text'):
                            hr_message = part.text
                            break

                # Trigger interrupt at orchestrator level
                interrupt_payload = {
                    "action": "hr_agent_approval",
                    "message": f"HR Agent needs approval:\n\n{hr_message}",
                    "task_id": task_id
                }

                logger.info("\n[ORCHESTRATOR HITL] Sending to frontend:")
                logger.info(json.dumps(interrupt_payload, indent=2))

                approval = interrupt(interrupt_payload)

                logger.info(f"\n[ORCHESTRATOR HITL] Received from frontend: '{approval}'\n")

                # Send approval back to HR agent
                if approval == "approved":
                    second_message_payload = {
                        'message': {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'approve'}],
                            'message_id': uuid4().hex,
                            'task_id': task_id,
                            'context_id': context_id,
                        },
                    }

                    second_request = SendMessageRequest(
                        id=str(uuid4()),
                        params=MessageSendParams(**second_message_payload)
                    )

                    logger.info("[ORCHESTRATOR] Sending approval to HR agent")
                    second_response = await client.send_message(second_request)

                    # Extract final result from the second response's status.message
                    result_text = "Task completed"
                    second_status = second_response.root.result.status
                    if hasattr(second_status, 'message') and second_status.message and hasattr(second_status.message, 'parts'):
                        for part in second_status.message.parts:
                            if hasattr(part, 'text'):
                                result_text = part.text
                                break

                    logger.info(f"[ORCHESTRATOR] HR agent completed with result")
                    response_msg = AIMessage(content=f"HR Agent: {result_text}")
                    return Command(goto="chat_node", update={"messages": [response_msg]})
                else:
                    # User rejected
                    cancel_msg = AIMessage(content="❌ HR agent request was cancelled by user.")
                    return Command(goto=END, update={"messages": [cancel_msg]})

            # Task completed without requiring approval
            elif task_state and (task_state.value == "completed" or str(task_state) == "TaskState.completed"):
                logger.info("[ORCHESTRATOR] HR agent task completed successfully")

                result_text = "No response from HR agent"
                if hasattr(task_status_obj, 'message') and task_status_obj.message and hasattr(task_status_obj.message, 'parts'):
                    for part in task_status_obj.message.parts:
                        if hasattr(part, 'text'):
                            result_text = part.text
                            break

                response_msg = AIMessage(content=f"HR Agent: {result_text}")
                return Command(goto="chat_node", update={"messages": [response_msg]})

            # Task failed
            elif task_state and (task_state.value == "failed" or str(task_state) == "TaskState.failed"):
                error = "Unknown error"
                if hasattr(task_status_obj, 'message'):
                    error = str(task_status_obj.message)
                error_msg = AIMessage(content=f"❌ HR agent task failed: {error}")
                return Command(goto=END, update={"messages": [error_msg]})

            return Command(goto=END)

        except GraphInterrupt:
            # GraphInterrupt is expected when calling interrupt() - let it propagate
            raise
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error calling HR agent: {e}", exc_info=True)
            error_msg = AIMessage(content=f"❌ Error communicating with HR agent: {str(e)}")
            return Command(goto=END, update={"messages": [error_msg]})


async def approval_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """Handle user approval for sensitive operations (HITL at orchestrator level)"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message:
        return Command(goto=END)

    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return Command(goto=END)

    # Check for weather tool call
    weather_call = next((tc for tc in tool_calls if tc.get("name") == "get_weather"), None)
    if weather_call:
        location = weather_call.get("args", {}).get("location", "unknown")

        interrupt_payload = {
            "action": "confirm_weather_request",
            "message": f"Do you approve fetching weather for {location}?",
            "location": location
        }

        logger.info("\n[ORCHESTRATOR HITL] Sending to frontend:")
        logger.info(json.dumps(interrupt_payload, indent=2))

        approval = interrupt(interrupt_payload)

        logger.info(f"\n[ORCHESTRATOR HITL] Received from frontend: '{approval}'\n")

        if approval == "approved":
            return Command(goto="tool_node")
        else:
            cancel_msg = AIMessage(content=f"❌ Weather request for {location} was cancelled by user.")
            return Command(goto=END, update={"messages": [cancel_msg]})

    # No recognized tool call requiring approval
    return Command(goto=END)


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
workflow.add_node("hr_agent_node", hr_agent_node)  # HR Agent as subgraph

# Define edges
workflow.add_edge("tool_node", "chat_node")
workflow.add_edge("hr_agent_node", "chat_node")  # Return to chat after HR agent completes
workflow.set_entry_point("chat_node")

# Compile the graph
# Note: Interrupts from hr_agent_node (subgraph) will automatically bubble up
graph = workflow.compile()

