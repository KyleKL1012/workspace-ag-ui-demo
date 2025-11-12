from pathlib import Path

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel

from travel_planner.helpers.general_utils import read_yaml


def read_yaml_and_parse_chat_prompt(yaml_path: Path) -> ChatPromptTemplate:
    """
    Reads a YAML file with 'system' and 'user' fields and converts them to a ChatPromptTemplate.
    """

    prompt_dict = read_yaml(yaml_path)
    system_prompt = SystemMessagePromptTemplate.from_template(
        prompt_dict["system_prompt"], template_format="jinja2"
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        prompt_dict["user_prompt"], template_format="jinja2"
    )

    return ChatPromptTemplate(messages=[system_prompt, user_prompt], template_format="jinja2")


class PromptTemplates(BaseModel):
    trip_params_extraction: ChatPromptTemplate
    fix_trip_params_extraction: ChatPromptTemplate
    trip_planner: ChatPromptTemplate
    routing_decision: ChatPromptTemplate
    chitchat_response: ChatPromptTemplate

    @classmethod
    def read_from_yaml(cls) -> "PromptTemplates":
        yaml_dir = Path(__file__).parent / "yaml"
        return cls(
            trip_params_extraction=read_yaml_and_parse_chat_prompt(
                yaml_dir / "trip_params_extraction.yaml"
            ),
            fix_trip_params_extraction=read_yaml_and_parse_chat_prompt(
                yaml_dir / "fix_trip_params_extraction.yaml"
            ),
            trip_planner=read_yaml_and_parse_chat_prompt(yaml_dir / "trip_planner.yaml"),
            routing_decision=read_yaml_and_parse_chat_prompt(yaml_dir / "routing_decision.yaml"),
            chitchat_response=read_yaml_and_parse_chat_prompt(yaml_dir / "chitchat_response.yaml"),
        )
