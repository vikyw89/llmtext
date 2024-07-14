import os
from typing import Annotated, AsyncIterable
from pydantic import BaseModel, Field
from llmtext.text import Text
from openai import AsyncOpenAI
import pytest


@pytest.mark.asyncio
async def test_arun():
    llm = Text(
        text="What is the capital of France ?",
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        ),
        openai_model="mistralai/mistral-7b-instruct-v0.3",
    )
    res = await llm.arun()
    assert res is not None


@pytest.mark.asyncio
async def test_stream():
    llm = Text(
        text="What is the capital of France ?",
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        ),
        openai_model="mistralai/mistral-7b-instruct-v0.3",
    )
    async for res in llm.astream():
        assert isinstance(res, str)


@pytest.mark.asyncio
async def test_structured_extraction():
    llm = Text(
        text="Paris is the capital of France.",
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        ),
        openai_model="mistralai/mistral-7b-instruct-v0.3",
    )

    class ExtractedData(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    res = await llm.astructured_extraction(
        output_class=ExtractedData,
    )

    assert isinstance(res, ExtractedData)


@pytest.mark.asyncio
async def test_astream_structured_extraction():
    llm = Text(
        text="Paris is the capital of France.",
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        ),
        openai_model="mistralai/mistral-7b-instruct-v0.3",
    )

    class City(BaseModel):
        name: Annotated[str, Field(description="Name of the city")]
        description: Annotated[str, Field(description="Description of the city")]

    class ExtractedData(BaseModel):
        cities: Annotated[list[City], Field(description="Cities")]

    res = await llm.astream_structured_extraction(
        output_class=ExtractedData,
    )

    assert isinstance(res, AsyncIterable)

    async for chunk in res:
        assert isinstance(chunk, ExtractedData)
        print(chunk.model_dump())
