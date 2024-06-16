import os
from typing import AsyncIterable
import instructor
from openai import AsyncOpenAI
from llmtext.chat_llms.base import BaseChatLLM
from llmtext.chat_llms.base import (
    T,
)


class ChatTogetherAI(BaseChatLLM):
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        max_retries: int = 2,
        client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("TOGETHERAI_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        ),
        *args,
        **kwargs
    ) -> None:
        super(self.__class__, self).__init__(*args, **kwargs)
        self.model = model
        self.max_retries = max_retries
        self.client = client
        self.structured_client = instructor.from_openai(self.client)

    async def arun(self) -> str:
        response = await self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
        )
        return response.choices[0].message.content or ""

    async def astream(self) -> AsyncIterable[str]:
        stream = await self.client.chat.completions.create(
            messages=self.messages, model=self.model, stream=True
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def astructured_extraction(self, output_class: type[T]) -> T:
        response = await self.structured_client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response
