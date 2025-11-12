import uuid
from typing import Any, AsyncGenerator, TypedDict

import reflex as rx
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
# from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from travel_planner.main import get_compiled_travel_planner_graph

# LangGraph travel planner imports
from travel_planner.models.state import TravelPlannerState

load_dotenv()


class QA(TypedDict):
    """A question and answer pair."""

    question: str
    answer: str


class UIState(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    _chats: dict[str, list[QA]] = {
        "Intros": [],
    }

    # The current chat name.
    current_chat = "Intros"

    # Whether we are processing the question.
    processing: bool = False

    # Whether the new chat modal is open.
    is_modal_open: bool = False

    # LangGraph compiled_graph (cached)
    _compiled_graph = None

    # Store interrupted conversations
    _interrupted_conversations: dict[str, dict] = {}

    # Langgraph Config
    _langgraph_config: RunnableConfig | None = None

    # Class-level storage for TravelPlannerState to avoid MutableProxy wrapping
    # This is stored at class level, not instance level, so Reflex won't wrap it
    _chat_states_storage: dict[str, TravelPlannerState] = {}

    def get_travel_planner_compiled_graph(self) -> CompiledStateGraph:
        """Get or create the travel planner compiled_graph using the main module function."""
        if self._compiled_graph is None:
            try:
                self._compiled_graph = get_compiled_travel_planner_graph()
            except Exception as e:
                print(f"Failed to initialize travel planner compiled_graph: {e}")
                self._compiled_graph = None
        return self._compiled_graph

    def _get_or_create_chat_state(self, question: str) -> TravelPlannerState:
        """Get existing chat state or create new one for this chat session."""
        # Use class-level storage to avoid MutableProxy wrapping
        if self.current_chat not in UIState._chat_states_storage:
            # Create new state for this chat session
            UIState._chat_states_storage[self.current_chat] = TravelPlannerState(
                user_prompt=question
            )
        else:
            # Update existing state with new prompt
            UIState._chat_states_storage[self.current_chat].user_prompt = question

        # Return the state directly from class storage (no MutableProxy)
        return UIState._chat_states_storage[self.current_chat]

    def _get_or_create_config(self) -> RunnableConfig:
        """Get existing config or create new one for this chat session."""
        if self.current_chat in self._interrupted_conversations:
            return self._interrupted_conversations[self.current_chat]["config"]
        else:
            if self._langgraph_config is None:
                # Create Langfuse Handler
                # langfuse_handler = CallbackHandler()
                # Create new config for new session
                thread_id = f"{self.current_chat}_{str(uuid.uuid4())[:12]}"
                self._langgraph_config = RunnableConfig(
                    configurable={"thread_id": thread_id},
                    # callbacks=[langfuse_handler],
                    # metadata={"langfuse_session_id": thread_id},
                )
            return self._langgraph_config

    def _is_continuation(self) -> bool:
        """Check if current chat has an interrupted conversation."""
        return self.current_chat in self._interrupted_conversations

    def _store_interruption(self, config: RunnableConfig, result: TravelPlannerState) -> None:
        """Store interrupted conversation state."""
        self._interrupted_conversations[self.current_chat] = {
            "config": config,
            "missing_params": result.missing_trip_params,
            "partial_result": result,
        }

    def _clear_interruption(self) -> None:
        """Clear interrupted conversation state."""
        if self.current_chat in self._interrupted_conversations:
            del self._interrupted_conversations[self.current_chat]

    @rx.event
    def create_chat(self, form_data: dict[str, Any]) -> None:
        """Create a new chat."""
        # Add the new chat to the list of chats.
        new_chat_name = form_data["new_chat_name"]
        self.current_chat = new_chat_name
        self._chats[new_chat_name] = []
        # Note: Don't create chat state here - it will be created on first message
        self.is_modal_open = False

    @rx.event
    def set_is_modal_open(self, is_open: bool) -> None:
        """Set the new chat modal open state.

        Args:
            is_open: Whether the modal is open.
        """
        self.is_modal_open = is_open

    @rx.var
    def selected_chat(self) -> list[QA]:
        """Get the list of questions and answers for the current chat.

        Returns:
            The list of questions and answers.
        """
        return self._chats[self.current_chat] if self.current_chat in self._chats else []

    @rx.event
    def delete_chat(self, chat_name: str) -> None:
        """Delete the current chat."""
        if chat_name not in self._chats:
            return
        del self._chats[chat_name]

        # Clean up associated state
        if chat_name in UIState._chat_states_storage:
            del UIState._chat_states_storage[chat_name]
        if chat_name in self._interrupted_conversations:
            del self._interrupted_conversations[chat_name]

        if len(self._chats) == 0:
            self._chats = {
                "Intros": [],
            }
        if self.current_chat not in self._chats:
            self.current_chat = list(self._chats.keys())[0]

    @rx.event
    def set_chat(self, chat_name: str) -> None:
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.event
    def set_new_chat_name(self, new_chat_name: str) -> None:
        """Set the name of the new chat.

        Args:
            new_chat_name: The name of the new chat.
        """
        self.new_chat_name = new_chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self._chats.keys())

    @rx.event
    async def process_question(self, form_data: dict[str, Any]) -> AsyncGenerator[None, None]:
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if not question:
            return

        # Route to LangGraph travel planner instead of direct OpenAI
        async for value in self.langgraph_process_question(question):
            yield value

    @rx.event
    async def langgraph_process_question(self, question: str) -> AsyncGenerator[None, None]:
        """Process the question using LangGraph travel planner."""
        # Add question to chat
        qa = QA(question=question, answer="")
        self._chats[self.current_chat].append(qa)
        self.processing = True
        yield

        try:
            compiled_graph = self.get_travel_planner_compiled_graph()
            if compiled_graph is None:
                self._chats[self.current_chat][-1]["answer"] = "⚠️ Travel planner unavailable."
                yield
                return

            # Get or create persistent state for this chat session
            travel_state = self._get_or_create_chat_state(question)
            config = self._get_or_create_config()

            # Simple if statement to handle continuation vs new conversation
            if self._is_continuation():
                # Get continued result directly without streaming
                travel_state.user_prompt = question  # Update prompt for continuation
                await compiled_graph.aupdate_state(config, travel_state)
                result = await compiled_graph.ainvoke(input=None, config=config)
                result = TravelPlannerState(**result)
                self._clear_interruption()

            else:
                # New conversation or first message with persistent state
                result = await compiled_graph.ainvoke(config=config, input=travel_state)
                result = TravelPlannerState(**result)

            # Update the stored state with the latest result
            if result:
                UIState._chat_states_storage[self.current_chat] = result

            # Use the response from LangGraph directly
            response = (
                result.last_ai_message if result and result.last_ai_message else "I'm here to help!"
            )

            # Check if interrupted (missing parameters)
            if result and result.missing_trip_params:
                self._store_interruption(config, result)

            self._chats[self.current_chat][-1]["answer"] = response
            yield

        except Exception as e:
            self._chats[self.current_chat][-1]["answer"] = f"⚠️ Error: {str(e)}"
            yield

        self.processing = False
