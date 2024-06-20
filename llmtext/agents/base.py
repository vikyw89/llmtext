from abc import abstractmethod
from typing import Annotated, Any, AsyncGenerator, AsyncIterable
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, Field
from llmtext.chat_llms.base import BaseChatLLM
from llmtext.chat_llms.openai import ChatOpenAI
from llmtext.llms.base import T
from typing import Awaitable, Callable
from llama_index.core.tools import BaseTool
from typing import get_type_hints


class ToolCall():
    def __init__(self, afn: Callable[[Any], Awaitable[Any]]) -> None:
        self.afn = afn
        self.name = afn.__name__
        self.input = get_type_hints(afn)["input"]
        self.output = get_type_hints(afn)["return"]
        self.description = afn.__doc__




class BaseAgent:
    def __init__(
        self,
        messages: list[ChatCompletionMessageParam] = [],
        async_tools: list[Callable[[Any], Awaitable[Any]]] = [],
        chat_llm: BaseChatLLM = ChatOpenAI(),
    ) -> None:
        self.messages = messages
        self.async_tools = async_tools
        self.steps = []
        self.chat_llm = chat_llm
        pass

    @abstractmethod
    async def arun(self) -> str:
        self.chat_llm.messages = self.messages

        # do structured_extraction of the tool
        tools = []
        for tool in self.async_tools:
            tools.append(ToolCall(afn=tool))

        class ToolSelect(BaseTool):
            """Selected tools, with their arguments"""
            chosen_tools = Annotated[list[ToolCall], Field(description="chosen tools, with their arguments")] = []


        step = await self.chat_llm.astream_structured_extraction(output_class=ToolSelect)
        pass

    @abstractmethod
    async def astream(self) -> AsyncGenerator[str]:
        pass

    @abstractmethod
    async def astructured_extraction(self, output_class: type[T]) -> T:
        pass

    @abstractmethod
    async def arun_step(self) -> str:
        pass

    @abstractmethod
    async def astream_step(self) -> AsyncGenerator[str]:
        pass

    @abstractmethod
    async def astructured_extraction_step(self, output_class: type[T]) -> T:
        pass

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        self.messages.append(message)
