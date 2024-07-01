import os
from typing import AsyncIterable
from typing import TypeVar
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Text:
    def __init__(
        self,
        text: str,
        openai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        ),
        openai_model: str = "gpt-3.5-turbo",
        togetherai_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        togetherai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("TOGETHERAI_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        ),
        openrouter_client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        ),
        openrouter_model: str = "anthropic/claude-3-haiku",
        max_retries: int = 3,
    ) -> None:
        self.text = text
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.togetherai_client = togetherai_client
        self.togetherai_model = togetherai_model
        self.openrouter_client = openrouter_client
        self.openrouter_model = openrouter_model
        self.max_retries = max_retries
        pass

    async def arun_openai(self) -> str:
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": self.text}],
        )

        return response.choices[0].message.content or ""

    async def arun_togetherai(self) -> str:
        response = await self.togetherai_client.chat.completions.create(
            model=self.togetherai_model,
            messages=[{"role": "user", "content": self.text}],
        )

        return response.choices[0].message.content or ""

    async def arun_openrouter(self) -> str:
        response = await self.openrouter_client.chat.completions.create(
            model=self.openrouter_model,
            messages=[{"role": "user", "content": self.text}],
        )

        return response.choices[0].message.content or ""

    async def astream_openai(
        self,
    ) -> AsyncIterable[str]:
        stream = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": self.text}],
            stream=True,
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astream_togetherai(
        self,
    ) -> AsyncIterable[str]:
        stream = await self.togetherai_client.chat.completions.create(
            model=self.togetherai_model,
            messages=[{"role": "user", "content": self.text}],
            stream=True,
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astream_openrouter(self) -> AsyncIterable[str]:
        stream = await self.openrouter_client.chat.completions.create(
            model=self.openrouter_model,
            messages=[{"role": "user", "content": self.text}],
            stream=True
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astructured_extraction_openai(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> T:
        structured_client = instructor.from_openai(self.openai_client)
        response = await structured_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": self.text},
            ],
            temperature=0,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response

    async def astructured_extraction_togetherai(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> T:
        structured_client = instructor.from_openai(self.togetherai_client)
        response = await structured_client.chat.completions.create(
            model=self.togetherai_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": self.text},
            ],
            temperature=0,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response

    async def astructured_extraction_openrouter(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> T:
        structured_client = instructor.from_openai(self.openrouter_client)
        response = await structured_client.chat.completions.create(
            model=self.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": self.text},
            ],
            temperature=0,
            max_retries=self.max_retries,
            response_model=output_class
        )
        return response
    
    async def astream_structured_extraction_openai(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> AsyncIterable[T]:
        structured_client = instructor.from_openai(self.openai_client)
        stream: AsyncIterable[output_class] = (
            structured_client.chat.completions.create_partial(
                model=self.openai_model,
                response_model=output_class,
                temperature=0,
                max_retries=self.max_retries,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {"role": "user", "content": self.text},
                ],
            )
        )

        return stream

    async def astream_structured_extraction_openrouter(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> AsyncIterable[T]:
        structured_client = instructor.from_openai(self.openrouter_client)
        stream: AsyncIterable[output_class] = (
            structured_client.chat.completions.create_partial(
                model=self.openrouter_model,
                response_model=output_class,
                max_retries=self.max_retries,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {"role": "user", "content": self.text},
                ],
            )
        )

        return stream