from abc import abstractmethod
from typing import Any
from pydantic import BaseModel


class RunnableTool(BaseModel):
    @abstractmethod
    async def arun(self) -> str:
        pass

    def to_context(self) -> dict[str, Any]:
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
