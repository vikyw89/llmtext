from abc import abstractmethod
from typing import Annotated, Any
from typing import Literal, TypedDict
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class Event(TypedDict):
    step: int
    type: Literal["tool_call", "tool_output", "message_stream", "message", "score"]
    id: str
    content: str


class RunnableTool(BaseModel):
    @abstractmethod
    async def arun(self) -> str:
        pass

    def get_tool_call(self) -> dict[str, Any]:
        tool_name = self.__class__.__name__
        return {
            "type": "tool_call",
            "name": tool_name,
            "description": self.__doc__ or "",
            "params": self.model_dump(),
        }

    async def aget_tool_output(self) -> dict[str, Any]:
        tool_name = self.__class__.__name__
        return {
            "type": "tool_output",
            "name": tool_name,
            "description": self.__doc__ or "",
            "params": self.model_dump(),
            "output": await self.arun(),
        }


class ToolSelector(BaseModel):
    """Selected tools"""

    choices: Annotated[
        list[RunnableTool], Field(description="Selected tool to call")
    ] = []
