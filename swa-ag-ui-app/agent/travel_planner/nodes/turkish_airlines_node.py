import os

from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from travel_planner.models.available_llm_models import LLMs
from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.base_node import BaseNode
from travel_planner.prompts.prompt_handler import PromptTemplates


class TurkishAirlinesNode(BaseNode):
    """
    Node that handles Turkish Airlines specific queries and integrates with
    Turkish Airlines MCP server for flight information, bookings, and services.
    """

    def __init__(self, prompt_templates: PromptTemplates, llm_models: LLMs):
        super().__init__()
        self.prompt_templates = prompt_templates
        self.llm_models = llm_models
        self.mcp_client = None
        self.turkish_airlines_agent = None

        # MCP configuration from environment variables
        self.node_bin_path = os.getenv("NODE_BIN_PATH", "/usr/local/bin")
        self.mcp_remote_command = os.getenv("MCP_REMOTE_COMMAND", "mcp-remote")

        # Log the configuration for debugging
        self.logger.info(
            f"Turkish Airlines MCP Config - Node path: "
            f"{self.node_bin_path}, MCP command: {self.mcp_remote_command}"
        )

    async def _initialize_mcp_client(self) -> None:
        """Initialize the MCP client and Turkish Airlines agent if not already done."""
        if self.mcp_client is None:
            try:
                env = os.environ.copy()
                # Ensure Node.js is available in PATH
                env["PATH"] = f"{self.node_bin_path}:{env.get('PATH', '')}"

                self.mcp_client = MultiServerMCPClient(
                    {
                        "turkish_airlines": {
                            "transport": "stdio",
                            "command": self.mcp_remote_command,
                            "args": [
                                "https://mcp.turkishtechlab.com/sse",
                                "--auth-timeout",
                                "120",
                            ],
                            "env": env,
                        }
                    }
                )

                tools = await self.mcp_client.get_tools()  # type: ignore
                self.logger.info(f"Loaded Turkish Airlines MCP tools: {[t.name for t in tools]}")

                # Create the agent with the large model and MCP tools
                self.turkish_airlines_agent = create_react_agent(self.llm_models.large_model, tools)
                self.logger.info("Turkish Airlines MCP agent initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize Turkish Airlines MCP client: {e}")
                raise

    async def async_run(self, state: TravelPlannerState) -> TravelPlannerState:  # type: ignore[override]
        try:
            # Initialize MCP client and agent if not already done
            await self._initialize_mcp_client()

            if self.turkish_airlines_agent is None:
                raise ValueError("Turkish Airlines MCP agent is not initialized")

            # Invoke the Turkish Airlines agent with the user's query
            self.logger.info(
                f"{self.node_id} | Processing Turkish Airlines query: {state.user_prompt}"
            )

            agent_response = await self.turkish_airlines_agent.ainvoke({"messages": state.messages})

            # Extract the response content
            if (
                agent_response
                and "messages" in agent_response
                and len(agent_response["messages"]) > 0
            ):
                ai_message = agent_response["messages"][-1]

                # Store the Turkish Airlines response in state
                state.last_ai_message = str(ai_message.content)
                state.messages.append(ai_message)

                self.logger.info(f"{self.node_id} | Successfully processed Turkish Airlines query")
            else:
                self.logger.error("Empty or invalid response from Turkish Airlines agent")
                # Fallback response
                fallback_message = (
                    "I'm sorry, I couldn't process your Turkish Airlines request at the moment. "
                    "Please try again later."
                )
                ai_message = AIMessage(content=fallback_message)
                state.last_ai_message = fallback_message
                state.messages.append(ai_message)

        except Exception as e:
            self.logger.error(f"Error in Turkish Airlines node: {e}")
            # Fallback response for errors
            error_message = (
                "I encountered an issue while processing your Turkish Airlines request. "
                "Please try again or contact support if the problem persists."
            )
            ai_message = AIMessage(content=error_message)
            state.last_ai_message = error_message
            state.messages.append(ai_message)

        return state
