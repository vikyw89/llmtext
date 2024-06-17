from typing import Annotated, AsyncGenerator
from pydantic import BaseModel, Field
from llmtext.chat_llms.openai import ChatOpenAI
import asyncio


def test_openai_arun():

    async def arun():

        llm = ChatOpenAI()
        llm.add_message({"role": "system", "content": "You are a helpful assistant."})
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        res = await llm.arun()
        assert res is not None

    asyncio.run(arun())


def test_openai_stream():

    async def astream():

        llm = ChatOpenAI()
        llm.add_message({"role": "system", "content": "You are a helpful assistant."})
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        stream = llm.astream()

        async for chunk in stream:
            assert isinstance(chunk, str)

    asyncio.run(astream())


def test_openai_structured_extraction():

    async def astructured_extraction():

        llm = ChatOpenAI()

        llm.add_message(
            {
                "role": "system",
                "content": "Extract what the user asks from the following conversations.",
            }
        )
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        llm.add_message(
            {"role": "assistant", "content": "The capital of France is Paris."}
        )
        llm.add_message({"role": "user", "content": "What is the capital of Germany?"})
        llm.add_message(
            {"role": "assistant", "content": "The capital of Germany is Berlin."}
        )

        class ExtractedData(BaseModel):
            questions: Annotated[
                list[str], Field(description="The questions asked by the user")
            ]

        res = await llm.astructured_extraction(output_class=ExtractedData)
        print("res", res)
        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


def test_chat_openai_astream_structured_extraction():

    llm = ChatOpenAI()

    class ExtractedData(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    async def arun():
        llm.messages = [
            {
                "role": "system",
                "content": "Extract what the user asks from the following conversations.",
            },
            {
                "role": "user",
                "content": "The capital of Germany is Berlin. It's a beautiful city. The capital of France is Paris. It's a beautiful city.",
            },
        ]

        res = await llm.astream_structured_extraction(
            output_class=ExtractedData,
        )

        assert isinstance(res, AsyncGenerator)

        async for chunk in res:
            print(chunk.model_dump())
            assert isinstance(chunk, ExtractedData)

    asyncio.run(arun())
