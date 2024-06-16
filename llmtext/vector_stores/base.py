from abc import abstractmethod


class BaseNode:
    def __init__(
        self,
        search_text: str,
        retrieve_text: str,
        metadata: dict = {},
        embedding: list[float] = [],
    ) -> None:
        self.search_text = search_text
        self.retrieve_text = retrieve_text
        self.metadata = metadata
        self.embedding = embedding
        pass

class BaseVectorStore:
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def aembed(self, text: str):
        pass

    @abstractmethod
    async def aretrieve(self, text: str) -> list[BaseNode]:
        pass

    @abstractmethod
    async def asearch(self, text: str) -> list[BaseNode]:
        pass
