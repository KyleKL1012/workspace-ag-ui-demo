from __future__ import annotations

from typing import TypeVar, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from pydantic import BaseModel

from travel_planner.models.available_llm_models import LLMs
from travel_planner.settings.settings_handler import GroqSettings

T = TypeVar("T", bound=BaseModel)


async def invoke_llm(
    prompt_value: PromptValue,
    llm: ChatGroq,
    response_model: type[T] | None = None,
    messages_history: list[BaseMessage] | None = None,
) -> T | BaseMessage | AIMessage:
    """
    Uses OpenAI function calling / JSON mode to parse directly into a Pydantic model.
    This requires LangChain >= 0.1.14+ which supports .with_structured_output().

    Args:
        chat_prompt_value: Rendered prompt to pass to the LLM.
        response_model:    The Pydantic model to parse into.
        llm:               A LangChain ChatOpenAI instance with JSON support.

    Returns:
        An instance of the response_model populated from the LLM.
    """
    messages = prompt_value.to_messages()
    system_message = messages[0]
    human_message = messages[1]
    if messages_history is None:
        messages_history = []
    input_messages: list[BaseMessage] = [system_message, *messages_history, human_message]
    if response_model is not None:
        structured_llm = llm.with_structured_output(response_model)
        result = await structured_llm.ainvoke(input=input_messages)
        return cast(T, result)
    else:
        # Explicitly bind an empty tools list to prevent tool calling when not needed
        llm_without_tools = llm.bind(tools=[])
        result = await llm_without_tools.ainvoke(input=input_messages)
        return result


def _create_models(settings: GroqSettings, model_name: str) -> ChatGroq:
    """
    Instantiates a ChatOpenAI object using typed OpenAISettings.

    Args:
        settings: OpenAISettings object with model config.

    Returns:
        A configured ChatOpenAI instance.
    """
    return ChatGroq(
        model=model_name,
        temperature=settings.temperature,
        timeout=settings.timeout,
    )


def get_available_llms(settings: GroqSettings) -> LLMs:
    return LLMs(
        large_model=_create_models(settings, settings.large_model),
        mini_model=_create_models(settings, settings.mini_model),
    )
