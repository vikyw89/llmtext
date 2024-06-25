from abc import abstractmethod
from typing import AsyncGenerator, AsyncIterable
from instructor.client import T
from openai.types.chat import (
    ChatCompletionMessageParam,
)


class BaseChatLLM:

    @abstractmethod
    async def arun(self) -> str:
        pass

    @abstractmethod
    async def astream(self) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def astructured_extraction(self, output_class: type[T]) -> T:
        pass

    @abstractmethod
    async def astream_structured_extraction(
        self, output_class: type[T]
    ) -> AsyncGenerator[T, None]:
        pass
