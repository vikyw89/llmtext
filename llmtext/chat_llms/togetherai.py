import os
from typing import AsyncGenerator
import instructor
from openai import AsyncOpenAI
from llmtext.chat_llms.openai import ChatOpenAI
from llmtext.llms.base import T


class ChatTogetherAI(ChatOpenAI):
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("TOGETHERAI_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        ),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = client
        self.structured_client = instructor.from_openai(self.client)

    # async def astream_structured_extraction(self, output_class: type[T]) -> AsyncGenerator[T, None]:
    #     raise NotImplementedError
