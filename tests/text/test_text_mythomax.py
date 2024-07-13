import os
from typing import Annotated, AsyncIterable
from pydantic import BaseModel, Field
from llmtext.text import Text
import asyncio
from openai import AsyncOpenAI


def test_arun():

    async def arun():

        llm = Text(
            text="What is the capital of France ?",
            openai_client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            ),
            openai_model="gryphe/mythomax-l2-13b",
        )
        res = await llm.arun()
        assert res is not None

    asyncio.run(arun())


def test_stream():

    async def astream():
        llm = Text(
            text="What is the capital of France ?",
            openai_client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            ),
            openai_model="gryphe/mythomax-l2-13b",
        )
        async for res in llm.astream():
            assert isinstance(res, str)

    asyncio.run(astream())


def test_structured_extraction():

    async def astructured_extraction():
        llm = Text(
            text="Paris is the capital of France.",
            openai_client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            ),
            openai_model="gryphe/mythomax-l2-13b",
        )

        class ExtractedData(BaseModel):
            name: Annotated[str, Field(description="Name of the city")]
            description: Annotated[str, Field(description="Description of the city")]

        res = await llm.astructured_extraction(
            output_class=ExtractedData,
        )

        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


def test_astream_structured_extraction():

    llm = Text(
        text="Paris is the capital of France.",
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        ),
        openai_model="gryphe/mythomax-l2-13b",
    )

    class City(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    class ExtractedData(BaseModel):
        cities: Annotated[list[City], Field(description="Cities")]

    async def arun():
        res = await llm.astream_structured_extraction(
            output_class=ExtractedData,
        )

        assert isinstance(res, AsyncIterable)

        async for chunk in res:
            assert isinstance(chunk, ExtractedData)
            print(chunk.model_dump())

    asyncio.run(arun())
