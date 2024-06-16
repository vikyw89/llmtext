import os
from typing import AsyncGenerator
from llmtext.llms.base import BaseLLM, T
from openai import AsyncOpenAI
import instructor


class TogetherAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        max_retries: int = 2,
        client: AsyncOpenAI = AsyncOpenAI(
            api_key=os.getenv("TOGETHERAI_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        ),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.max_retries = max_retries
        self.client = client
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
