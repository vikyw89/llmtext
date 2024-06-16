from abc import abstractmethod
from typing import AsyncIterable
from llmtext.llms.base import T


class BaseLongContextLLM:
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def arun(self, text: str) -> str:
        pass

    @abstractmethod
    async def astructured_extraction(self, text: str, output_class: type[T]) -> T:
        pass

    @abstractmethod
    async def astream(self, text: str) -> AsyncIterable[str]:
        pass
