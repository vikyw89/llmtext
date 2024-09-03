from abc import abstractmethod
from typing import Annotated, Any
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class ToolOutput(TypedDict):
    type: Literal["tool_output"]
    name: str
    description: str
    params: dict[str, Any]
    output: str


class ToolCall(TypedDict):
    type: Literal["tool_call"]
    name: str
    description: str
    params: dict[str, Any]


class Evaluation(TypedDict):
    type: Literal["evaluation"]
    is_final: bool


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class Checkpoint(TypedDict):
    type: Literal["checkpoint"]
    messages: list[ChatCompletionMessageParam]


class Event(TypedDict):
    step: int
    type: Literal[
        "tool_call",
        "tool_output",
        "message_stream",
        "message",
        "evaluation",
        "checkpoint",
    ]
    id: str
    content: ToolCall | ToolOutput | Message | Evaluation | Checkpoint


class RunnableTool(BaseModel):
    @abstractmethod
    async def _arun(self) -> str:
        pass

    def to_tool_call(self) -> ToolCall:
        tool_name = self.__class__.__name__
        return {
            "type": "tool_call",
            "name": tool_name,
            "description": self.__doc__ or "",
            "params": self.model_dump(),
        }

    async def acall_and_return_tool_output(self) -> ToolOutput:
        tool_name = self.__class__.__name__
        return {
            "type": "tool_output",
            "name": tool_name,
            "description": self.__doc__ or "",
            "params": self.model_dump(),
            "output": await self._arun(),
        }


class ToolSelector(BaseModel):
    """Selected tools"""

    choices: Annotated[
        list[RunnableTool], Field(description="Selected tool to call")
    ] = []


class IsFinalResponse(BaseModel):
    """Is the response final or not"""

    is_final_response: Annotated[
        bool,
        Field(
            description="True if the response is final, False if the response is not final",
            default=True,
        ),
    ]
