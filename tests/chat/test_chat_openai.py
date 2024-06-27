from typing import Annotated, AsyncGenerator
from pydantic import BaseModel, Field
from llmtext.chat.index import Chat
import asyncio
from openai.types.chat import ChatCompletionMessageParam


def test_openai_arun():

    async def arun():
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        llm = Chat(messages=messages)

        res = await llm.arun_openai()
        assert res is not None

    asyncio.run(arun())


def test_openai_stream():

    async def astream():

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        llm = Chat(messages=messages)
        stream = llm.astream_openai()

        async for chunk in stream:
            # assert isinstance(chunk.get("content"), str)
            print("chunk", chunk)

    asyncio.run(astream())


def test_openai_structured_extraction():

    async def astructured_extraction():

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "Extract what the user asks from the following conversations.",
            },
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
        ]
        llm = Chat(messages=messages)

        class ExtractedData(BaseModel):
            questions: Annotated[
                list[str], Field(description="The questions asked by the user")
            ]

        res = await llm.astructured_extraction_openai(output_class=ExtractedData)
        print("res", res)
        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


def test_chat_openai_astream_structured_extraction():

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "Extract what the user asks from the following conversations.",
        },
        {
            "role": "user",
            "content": "Generate cities and descriptions",
        },
    ]
    llm = Chat(messages=messages)

    class ExtractedData(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    async def arun():

        res = await llm.astream_structured_extraction_openai(output_class=ExtractedData)

        assert isinstance(res, AsyncGenerator)

        async for chunk in res:
            print(chunk.model_dump())
            assert isinstance(chunk, ExtractedData)

    asyncio.run(arun())
