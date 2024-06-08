from abc import abstractmethod
from openai import AsyncStream
from instructor.client import T
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessage,
    ChatCompletionChunk,
)


class BaseChatLLM:
    def __init__(
        self,
        messages: list[ChatCompletionMessageParam] = [],
    ) -> None:
        self.messages = messages

    @abstractmethod
    async def arun(self) -> ChatCompletionMessage:
        pass

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        self.messages.append(message)

    @abstractmethod
    async def astream(self) -> AsyncStream[ChatCompletionChunk]:
        pass

    @abstractmethod
    async def astructured_extraction(self, output_class: type[T]) -> T:
        pass
