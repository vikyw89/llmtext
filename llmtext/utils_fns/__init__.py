from typing import Type, Union, Annotated
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from llmtext.data_types import Message, RunnableTool


def messages_to_openai_messages(
    messages: list[Message],
) -> list[ChatCompletionMessageParam]:
    parsed_messages = []
    for message in messages:
        parsed_messages.append({"role": message.role, "content": message.content})
    return parsed_messages


def tools_to_tool_selector(
    tools: list[Type[RunnableTool]],
    prompt: str = "Selected tools to call in order to give the best response to user's query",
):
    tuple_tools = tuple(tools)
    tools = list[Union[*tuple_tools]]  # type: ignore

    class ToolSelector(BaseModel):
        f"""{prompt}"""

        tool_calls: Annotated[tools, Field(description="Tools to call")] = []  # type: ignore

    return ToolSelector
