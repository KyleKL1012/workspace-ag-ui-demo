import reflex as rx
from reflex.constants.colors import ColorType

from travel_planner.ui.chat.ui_state import QA, UIState


def message_content(text: str, color: ColorType) -> rx.Component:
    """Create a message content component.

    Args:
        text: The text to display.
        color: The color of the message.

    Returns:
        A component displaying the message.
    """
    return rx.markdown(
        text,
        background_color=rx.color(color, 4),
        color=rx.color(color, 12),
        display="inline-block",
        padding_inline="1em",
        border_radius="8px",
    )


def message(qa: QA) -> rx.Component:
    """A single question/answer message.

    Args:
        qa: The question/answer pair.

    Returns:
        A component displaying the question/answer pair.
    """
    return rx.box(
        rx.box(
            message_content(qa["question"], "mauve"),
            text_align="right",
            margin_bottom="8px",
        ),
        rx.box(
            message_content(qa["answer"], "accent"),
            text_align="left",
            margin_bottom="8px",
        ),
        max_width="50em",
        margin_inline="auto",
    )


def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.auto_scroll(
        rx.foreach(UIState.selected_chat, message),
        flex="1",
        padding="8px",
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""
    return rx.center(
        rx.vstack(
            rx.form(
                rx.hstack(
                    rx.input(
                        rx.input.slot(
                            rx.tooltip(
                                rx.icon("info", size=18),
                                content="Enter a question to get a response.",
                            )
                        ),
                        placeholder="Type something...",
                        id="question",
                        flex="1",
                    ),
                    rx.button(
                        "Send",
                        loading=UIState.processing,
                        disabled=UIState.processing,
                        type="submit",
                    ),
                    max_width="50em",
                    margin="0 auto",
                    align_items="center",
                ),
                reset_on_submit=True,
                on_submit=UIState.process_question,
            ),
            width="100%",
            padding_x="16px",
            align="stretch",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align="stretch",
        width="100%",
    )
