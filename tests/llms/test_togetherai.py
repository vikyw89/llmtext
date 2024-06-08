from typing import Annotated
from pydantic import BaseModel, Field


def test_together_ai_arun():
    import asyncio

    async def arun():
        from llmtext.llms.togetherai import TogetherAILLM

        llm = TogetherAILLM()
        res = await llm.arun("What is the capital of France?")
        assert res is not None

    asyncio.run(arun())

def test_together_ai_stream():
    import asyncio

    async def astream():
        from llmtext.llms.togetherai import TogetherAILLM

        llm = TogetherAILLM()
        async for res in llm.astream("What is the capital of France?"):
            assert isinstance(res, str) 

    asyncio.run(astream())


def test_together_ai_structured_extraction():
    import asyncio

    async def astructured_extraction():
        from llmtext.llms.togetherai import TogetherAILLM

        llm = TogetherAILLM()

        class ExtractedData(BaseModel):
            name: Annotated[str, Field(description="Name of the city")]
            description: Annotated[str, Field(description="Description of the city")]

        res = await llm.astructured_extraction(text="The city of France is Paris. It's a beautiful city.", output_class=ExtractedData)

        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())