from typing import Annotated
from pydantic import BaseModel, Field
from llmtext.llms.togetherai import TogetherAILLM
import asyncio


def test_together_ai_arun():

    async def arun():

        llm = TogetherAILLM()
        res = await llm.arun("What is the capital of France?")
        assert res is not None

    asyncio.run(arun())


def test_together_ai_stream():

    async def astream():

        llm = TogetherAILLM()
        async for res in llm.astream("What is the capital of France?"):
            assert isinstance(res, str)

    asyncio.run(astream())


def test_together_ai_structured_extraction():

    async def astructured_extraction():

        llm = TogetherAILLM()

        class ExtractedData(BaseModel):
            name: Annotated[str, Field(description="Name of the city")]
            description: Annotated[str, Field(description="Description of the city")]

        res = await llm.astructured_extraction(
            text="The city of France is Paris. It's a beautiful city.",
            output_class=ExtractedData,
        )

        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


# def test_togetherai_astream_structured_extraction():

#     llm = TogetherAILLM()

#     class ExtractedData(BaseModel):
#         name: Annotated[str, Field(description="Name of the city")]
#         description: Annotated[str, Field(description="Description of the city")]

#     async def arun():
#         res = await llm.astream_structured_extraction(
#             text="The city of France is Paris. It's a beautiful city. The city of Philippines is Manila. It's a beautiful city.",
#             output_class=ExtractedData,
#         )
#         assert isinstance(res, AsyncIterable)
#         async for chunk in res:
#             print(chunk)
#             assert isinstance(chunk, ExtractedData)

#     asyncio.run(arun())
