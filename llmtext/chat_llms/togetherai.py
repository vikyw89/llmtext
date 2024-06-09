import os
import instructor
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from llmtext.chat_llms.base import BaseChatLLM
from llmtext.chat_llms.base import (
    ChatCompletionMessage,
    ChatCompletionChunk,
    T,
)


class ChatTogetherAI(BaseChatLLM):
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        api_key: str = os.getenv("TOGETHERAI_API_KEY", "") or "",
        tools: list[ChatCompletionToolParam] = [],
        max_retries: int = 2,
        base_url: str = "https://api.together.xyz/v1",
        *args,
        **kwargs
    ) -> None:
        super(self.__class__, self).__init__(*args, **kwargs)
        self.model = model
        self.api_key = api_key
        self.tools = tools if tools is not None else NOT_GIVEN
        self.max_retries = max_retries
        self.base_url = base_url
        self.client = AsyncOpenAI(
            api_key=self.api_key, max_retries=self.max_retries, base_url=self.base_url
        )
        self.structured_client = instructor.from_openai(self.client)

    async def arun(self) -> ChatCompletionMessage:
        response = await self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            tools=self.tools,
        )
        return response.choices[0].message

    async def astream(self) -> AsyncStream[ChatCompletionChunk]:
        stream = await self.client.chat.completions.create(
            messages=self.messages, model=self.model, stream=True, tools=self.tools
        )
        return stream

    async def astructured_extraction(self, output_class: type[T]) -> T:
        response = await self.structured_client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            max_retries=self.max_retries,
            response_model=output_class,
        )
        return response
