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
        super(self.__class__, self).__init__(*args, **kwargs)
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.client = AsyncOpenAI(api_key=self.api_key, max_retries=max_retries)
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

    def _get_token_count(self, text: str) -> int:
        import tiktoken

        encoding = tiktoken.encoding_for_model(self.model)
        embeddings = encoding.encode(text=text)

        return len(embeddings)

    def _split_text_by_line(self, text: str, max_tokens: int) -> list[str]:
        raw_lines = text.splitlines()
        lines = []
        for line in raw_lines:
            if self._get_token_count(text=line) < max_tokens:
                lines.append(line)
                continue

            # if the line is too long, split it by sentences
            sentences = self._split_text_by_sentence(text=line)
            merged_sentences = self._merge_chunks(
                lines=sentences, max_tokens=max_tokens
            )
            lines += merged_sentences
        return text.split("\n")

    def _merge_chunks(self, lines: list[str], max_tokens: int) -> list[str]:
        final = []
        current_token_count = 0
        for line in lines:
            line_token_count = self._get_token_count(text=line)

            # if final is empty, merge it
            if len(final) == 0:
                final.append(line)
                current_token_count = line_token_count
                continue

            # if final is not empty, check if the combined tokens are less than max_tokens
            if line_token_count + current_token_count < max_tokens:
                final[-1] += f"\n{line}"
                current_token_count += line_token_count
                continue

            # if the combined tokens are greater than max_tokens, split the line
            final.append(line)
            current_token_count = line_token_count

        return final

    def _chunk_text_by_line(self, text: str, max_tokens: int) -> list[str]:
        lines = self._split_text_by_line(text=text, max_tokens=max_tokens)
        return self._merge_chunks(lines=lines, max_tokens=max_tokens)
