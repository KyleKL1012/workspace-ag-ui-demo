from pathlib import Path

from pydantic import BaseModel, Field

from travel_planner.helpers.general_utils import read_yaml


class GroqSettings(BaseModel):
    large_model: str
    mini_model: str
    temperature: float = 0.0
    timeout: int | None = Field(default=15, description="Timeout in seconds")

# class OpenAISettings(BaseModel):
#     large_model: str
#     mini_model: str
#     temperature: float = 0.0
#     timeout: int | None = Field(default=15, description="Timeout in seconds")


# class AppSettings(BaseModel):
#     openai: OpenAISettings

#     @classmethod
#     def read_from_yaml(cls) -> "AppSettings":
#         """
#         Loads YAML config and parses it into the AppSettings model.

#         Args:
#             yaml_path: Path to the settings.yaml file

#         Returns:
#             AppSettings instance with nested config
#         """
#         yaml_path = Path(__file__).parent / "yaml"
#         return cls(
#             openai=OpenAISettings(**read_yaml(yaml_path / "openai_settings.yaml")),
#         )
    
class AppSettings(BaseModel):
    groq: GroqSettings

    @classmethod
    def read_from_yaml(cls) -> "AppSettings":
        """
        Loads YAML config and parses it into the AppSettings model.

        Args:
            yaml_path: Path to the settings.yaml file

        Returns:
            AppSettings instance with nested config
        """
        yaml_path = Path(__file__).parent / "yaml"
        return cls(
            groq=GroqSettings(**read_yaml(yaml_path / "groq_settings.yaml")),
        )
