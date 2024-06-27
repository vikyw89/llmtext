from abc import abstractmethod
import os
from typing import AsyncGenerator, Sequence
import instructor
from instructor.client import T
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class Chat:
    def __init__(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        openai_model: str = "gpt-3.5-turbo",
        openai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        ),
        togetherai_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        togetherai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("TOGETHERAI_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        ),
        max_retries: int = 3,
    ) -> None:
        self.messages = messages
        self.openai_model = openai_model
        self.openai_client = openai_client
        self.togetherai_model = togetherai_model
        self.togetherai_client = togetherai_client
        self.max_retries = max_retries
        pass

    async def arun_openai(self) -> str:
        response = await self.openai_client.chat.completions.create(
            messages=self.messages,
            model=self.openai_model,
        )
        message = response.choices[0].message.content
        return message or ""

    async def arun_togetherai(self) -> str:
        response = await self.togetherai_client.chat.completions.create(
            messages=self.messages,
            model=self.togetherai_model,
        )
        message = response.choices[0].message.content
        return message or ""

    async def astream_openai(self) -> AsyncGenerator[str, None]:
        stream = await self.openai_client.chat.completions.create(
            messages=self.messages, model=self.openai_model, stream=True
        )

        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    async def astream_togetherai(self) -> AsyncGenerator[str, None]:
        stream = await self.togetherai_client.chat.completions.create(
            messages=self.messages, model=self.togetherai_model, stream=True
        )

        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    async def astructured_extraction_openai(self, output_class: type[T]) -> T:
        structured_client = instructor.from_openai(self.openai_client)
        response = await structured_client.chat.completions.create(
            messages=self.messages,  # type: ignore
            model=self.openai_model,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response

    async def astructured_extraction_togetherai(self, output_class: type[T]) -> T:
        structured_client = instructor.from_openai(self.togetherai_client)
        response = await structured_client.chat.completions.create(
            messages=self.messages,  # type: ignore
            model=self.togetherai_model,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response

    async def astream_structured_extraction_openai(
        self, output_class: type[T]
    ) -> AsyncGenerator[T, None]:
        structured_client = instructor.from_openai(self.openai_client)
        stream: AsyncGenerator[output_class, None] = (
            structured_client.chat.completions.create_partial(
                model=self.openai_model,
                response_model=output_class,
                max_retries=self.max_retries,
                messages=self.messages,  # type: ignore
            )
        )
        return stream
