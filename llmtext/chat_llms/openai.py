import os
from typing import AsyncGenerator
import instructor
from openai import AsyncOpenAI
from llmtext.chat_llms.base import BaseChatLLM, T
from openai.types.chat import (
    ChatCompletionMessageParam,
)


class ChatOpenAI(BaseChatLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "")),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.max_retries = max_retries
        self.client = client
        self.structured_client = instructor.from_openai(self.client)

    async def arun(self, messages: list[ChatCompletionMessageParam]) -> str:
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
        )
        message = response.choices[0].message.content
        return message or ""

    async def astream(
        self, messages: list[ChatCompletionMessageParam]
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            messages=messages, model=self.model, stream=True
        )

        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    async def astructured_extraction(
        self, messages: list[ChatCompletionMessageParam], output_class: type[T]
    ) -> T:
        response = await self.structured_client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response

    async def astream_structured_extraction(
        self, messages: list[ChatCompletionMessageParam], output_class: type[T]
    ) -> AsyncGenerator[T, None]:
        stream: AsyncGenerator[output_class, None] = (
            self.structured_client.chat.completions.create_partial(
                model=self.model,
                response_model=output_class,
                max_retries=self.max_retries,
                messages=messages,
            )
        )
        return stream
