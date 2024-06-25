from typing import Annotated, AsyncGenerator
from pydantic import BaseModel, Field
from llmtext.chat_llms.openai import ChatOpenAI
import asyncio


def test_openai_arun():

    async def arun():

        llm = ChatOpenAI()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        res = await llm.arun(messages=messages)
        assert res is not None

    asyncio.run(arun())


def test_openai_stream():

    async def astream():

        llm = ChatOpenAI()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "What is the capital of France?",
            },
        ]
        stream = llm.astream(messages=messages)

        async for chunk in stream:
            # assert isinstance(chunk.get("content"), str)
            print("chunk", chunk)

    asyncio.run(astream())


def test_openai_structured_extraction():

    async def astructured_extraction():

        llm = ChatOpenAI()

        messages = [
            {
                "role": "system",
                "content": "Extract what the user asks from the following conversations.",
            },
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
        ]

        class ExtractedData(BaseModel):
            questions: Annotated[
                list[str], Field(description="The questions asked by the user")
            ]

        res = await llm.astructured_extraction(
            output_class=ExtractedData, messages=messages
        )
        print("res", res)
        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


def test_chat_openai_astream_structured_extraction():

    llm = ChatOpenAI()

    class ExtractedData(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    async def arun():
        messages = [
            {
                "role": "system",
                "content": "Extract what the user asks from the following conversations.",
            },
            {
                "role": "user",
                "content": "Generate cities and descriptions",
            },
        ]

        res = await llm.astream_structured_extraction(
            output_class=ExtractedData, messages=messages
        )

        assert isinstance(res, AsyncGenerator)

        async for chunk in res:
            print(chunk.model_dump())
            assert isinstance(chunk, ExtractedData)

    asyncio.run(arun())
