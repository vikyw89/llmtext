import os
from typing import AsyncIterable, Iterable
from llmtext.llms.base import BaseLLM, T
from openai import AsyncOpenAI
import instructor


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "")),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.max_retries = max_retries
        self.client = client
        self.structured_client = instructor.from_openai(self.client)

    async def astream(self, text: str) -> AsyncIterable[str]:
        stream = await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": text}], stream=True
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astructured_extraction(
        self,
        text: str,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> T:
        response = await self.structured_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": text},
            ],
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response

    async def arun(self, text: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )

        return response.choices[0].message.content or ""

    async def astream_structured_extraction(
        self,
        text: str,
        output_class: type[T],
        prompt: str = "Let's think step by step. Given a text, extract structured data from it.",
    ) -> AsyncIterable[T]:
        stream: AsyncIterable[output_class] = (
            await self.structured_client.chat.completions.create(
                model=self.model,
                response_model=Iterable[output_class],
                max_retries=self.max_retries,
                stream=True,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {"role": "user", "content": text},
                ],
            )
        )

        return stream
