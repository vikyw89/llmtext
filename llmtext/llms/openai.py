import os
from typing import AsyncGenerator
from llmtext.llms.base import BaseLLM, T
from openai import AsyncOpenAI
import instructor


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        max_retries: int = 2,
        max_input_tokens: int = 12000,
        max_output_tokens: int = 4000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.client = AsyncOpenAI(api_key=self.api_key, max_retries=self.max_retries)
        self.structured_client = instructor.from_openai(self.client)

    async def arun(self, text: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )

        return response.choices[0].message.content or ""

    async def astream(self, text: str) -> AsyncGenerator[str, None]:
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
