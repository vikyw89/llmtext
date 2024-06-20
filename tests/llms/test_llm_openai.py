from typing import Annotated, AsyncIterable
from pydantic import BaseModel, Field
from llmtext.llms.openai import OpenAILLM
import asyncio


def test_openai_arun():

    async def arun():

        llm = OpenAILLM()
        res = await llm.arun(text="What is the capital of France?")
        assert res is not None

    asyncio.run(arun())


def test_openai_stream():

    async def astream():

        llm = OpenAILLM()

        async for res in llm.astream("What is the capital of France?"):
            assert isinstance(res, str)

    asyncio.run(astream())


def test_openai_structured_extraction():

    async def astructured_extraction():

        llm = OpenAILLM()

        class ExtractedData(BaseModel):
            name: Annotated[str, Field(description="Name of the city")]
            description: Annotated[str, Field(description="Description of the city")]

        res = await llm.astructured_extraction(
            text="The city of France is Paris. It's a beautiful city.",
            output_class=ExtractedData,
        )

        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


def test_openai_astream_structured_extraction():

    llm = OpenAILLM()

    class ExtractedData(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    async def arun():
        res = await llm.astream_structured_extraction(
            text="The city of France is Paris. It's a beautiful city. The city of Philippines is Manila. It's a beautiful city.",
            output_class=ExtractedData,
        )

        assert isinstance(res, AsyncIterable)

        async for chunk in res:
            assert isinstance(chunk, ExtractedData)
            print(chunk.model_dump())

    asyncio.run(arun())
