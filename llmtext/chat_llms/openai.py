import os
import instructor
from openai import AsyncOpenAI, AsyncStream, NOT_GIVEN
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from llmtext.chat_llms.base import BaseChatLLM
from llmtext.chat_llms.base import (
    ChatCompletionMessage,
    ChatCompletionChunk,
    T,
)


class ChatOpenAI(BaseChatLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        tools: list[ChatCompletionToolParam] | None = None,
        *args,
        **kwargs
    ) -> None:
        super(self.__class__, self).__init__(*args, **kwargs)
        self.model = model
        self.api_key = api_key
        self.tools = tools if tools is not None else NOT_GIVEN

        self.client = AsyncOpenAI(api_key=self.api_key)
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
            max_retries=3,
            response_model=output_class,
        )
        return response
