# llm is a class that takes a text and returns a response text
from abc import abstractmethod
from typing import AsyncGenerator
from typing import TypeVar
from pydantic import BaseModel
import nltk

nltk.download("punkt")

T = TypeVar("T", bound=BaseModel)


class BaseLLM:
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def arun(self, text: str) -> str:
        pass

    @abstractmethod
    async def astream(self, text: str) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def astructured_extraction(self, text: str, output_class: T) -> T:
        pass


