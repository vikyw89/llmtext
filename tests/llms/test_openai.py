from typing import Annotated
from pydantic import BaseModel, Field


def test_openai_arun():
    import asyncio

    async def arun():
        from llmtext.llms.openai import OpenaiLLM

        llm = OpenaiLLM()
        res = await llm.arun(text="What is the capital of France?")
        assert res is not None

    asyncio.run(arun())

def test_openai_stream():
    import asyncio

    async def astream():
        from llmtext.llms.openai import OpenaiLLM

        llm = OpenaiLLM()
        async for res in llm.astream("What is the capital of France?"):
            assert isinstance(res, str) 

    asyncio.run(astream())


def test_openai_structured_extraction():
    import asyncio

    async def astructured_extraction():
        from llmtext.llms.openai import OpenaiLLM

        llm = OpenaiLLM()

        class ExtractedData(BaseModel):
            name: Annotated[str, Field(description="Name of the city")]
            description: Annotated[str, Field(description="Description of the city")]

        res = await llm.astructured_extraction(text="The city of France is Paris. It's a beautiful city.", output_class=ExtractedData)

        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())