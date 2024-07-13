import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import AsyncIterable
from typing import TypeVar
import instructor

T = TypeVar("T", bound=BaseModel)


class Text:
    def __init__(
        self,
        text: str,
        openai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        ),
        openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    ) -> None:
        self.text = text
        self.openai_client = openai_client
        self.openai_model = openai_model
        pass

    async def arun(self, **kwargs) -> str:
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": self.text}],
            **kwargs
        )

        return response.choices[0].message.content or ""

    async def astream(self, **kwargs) -> AsyncIterable[str]:
        stream = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": self.text}],
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astructured_extraction(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
        max_retries: int = 3,
        temperature: float = 0.0,
        instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
        **kwargs
    ) -> T:
        structured_client = instructor.from_openai(
            self.openai_client, mode=instructor_mode
        )
        response = await structured_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": self.text},
            ],
            temperature=temperature,
            max_retries=max_retries,
            response_model=output_class,
            **kwargs
        )
        return response

    async def astream_structured_extraction(
        self,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
        max_retries: int = 3,
        temperature: float = 0.0,
        instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
        **kwargs
    ) -> AsyncIterable[T]:
        structured_client = instructor.from_openai(
            self.openai_client, mode=instructor_mode
        )
        stream: AsyncIterable[output_class] = (
            structured_client.chat.completions.create_partial(
                model=self.openai_model,
                response_model=output_class,
                max_retries=max_retries,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {"role": "user", "content": self.text},
                ],
                **kwargs
            )
        )

        return stream
