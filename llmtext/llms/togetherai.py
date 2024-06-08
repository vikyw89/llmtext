import os
from typing import AsyncGenerator
from llmtext.llms.base import BaseLLM, T
from openai import AsyncOpenAI
import instructor


class TogetherAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        api_key: str = os.getenv("TOGETHERAI_API_KEY", ""),
        max_retries: int = 2,
        base_url: str = "https://api.together.xyz/v1",
        *args,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__(*args, **kwargs)
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=api_key, max_retries=max_retries, base_url=base_url
        )
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