from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import AsyncGenerator, Iterable, Type
import os
from openai import AsyncOpenAI
from typing import TypeVar
import instructor
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(
        self,
        client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
        ),
        model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.instructor_mode = instructor_mode
        self.kwargs = kwargs

        self.structured_client = instructor.from_openai(client, mode=instructor_mode)

    async def agenerate_response_from_text(self, text: str) -> str:
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model=self.model,
            **self.kwargs,
        )
        return response.choices[0].message.content

    async def astream_response_from_text(self, text: str) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model=self.model,
            stream=True,
            **self.kwargs,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def agenerate_response_from_messages(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> str:
        stream = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            **self.kwargs,
        )

        final_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                final_response += chunk.choices[0].delta.content

        return final_response

    async def astream_response_from_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            **self.kwargs,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astructured_extraction_from_text(
        self,
        text: str,
        output_class: Type[T],
    ) -> T:
        patched_client = instructor.from_openai(
            self.client,
            mode=self.instructor_mode,
        )
        completion = await patched_client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model=self.model,
            response_model=output_class,
            **self.kwargs,
        )

        return completion

    async def astructured_extraction_from_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        output_class: Type[T],
    ) -> T:
        completion = await self.structured_client.chat.completions.create(
            messages=messages,
            model=self.model,
            response_model=output_class,
            **self.kwargs,
        )

        return completion

    async def astream_structured_extraction_from_text(
        self,
        text: str,
        output_class: Type[T],
    ) -> AsyncGenerator[T, None]:
        stream = self.structured_client.chat.completions.create_partial(
            model=self.model,
            response_model=output_class,
            messages=[{"role": "user", "content": text}],
            stream=True,
            **self.kwargs,
        )
        return stream

    async def astream_structured_extraction_from_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        output_class: Type[T],
    ) -> AsyncGenerator[T, None]:
        stream = self.structured_client.chat.completions.create_partial(
            model=self.model,
            response_model=output_class,
            messages=messages,
            stream=True,
            **self.kwargs,
        )
        return stream
