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
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        ),
    ) -> None:
        self.messages = messages
        self.openai_model = openai_model
        self.openai_client = openai_client
        pass

    async def arun(self, **kwargs) -> str:
        response = await self.openai_client.chat.completions.create(
            messages=self.messages,
            model=self.openai_model,
            **kwargs,
        )
        message = response.choices[0].message.content
        return message or ""

    async def astream(self, **kwargs) -> AsyncGenerator[str, None]:
        stream = await self.openai_client.chat.completions.create(
            messages=self.messages, model=self.openai_model, stream=True, **kwargs
        )

        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    async def astructured_extraction(
        self,
        output_class: type[T],
        max_retries: int = 3,
        temperature: float = 0.0,
        instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
        **kwargs
    ) -> T:
        structured_client = instructor.from_openai(
            self.openai_client, mode=instructor_mode
        )
        response = await structured_client.chat.completions.create(
            messages=self.messages,  # type: ignore
            model=self.openai_model,
            max_retries=max_retries,
            temperature=temperature,
            response_model=output_class,
            **kwargs,
        )
        return response

    async def astream_structured_extraction(
        self,
        output_class: type[T],
        max_retries: int = 3,
        temperature: float = 0.0,
        instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
        **kwargs
    ) -> AsyncGenerator[T, None]:
        structured_client = instructor.from_openai(
            self.openai_client, mode=instructor_mode
        )
        stream: AsyncGenerator[output_class, None] = (
            structured_client.chat.completions.create_partial(
                model=self.openai_model,
                response_model=output_class,
                temperature=temperature,
                max_retries=max_retries,
                messages=self.messages,  # type: ignore
                stream=True,
                **kwargs,
            )
        )
        return stream
