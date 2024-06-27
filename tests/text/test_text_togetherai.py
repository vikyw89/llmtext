from typing import Annotated
from pydantic import BaseModel, Field
from llmtext.text.index import Text
import asyncio


def test_together_ai_arun():

    async def arun():

        llm = Text(text="What is the capital of France ?")
        res = await llm.arun_togetherai()
        assert res is not None

    asyncio.run(arun())


def test_together_ai_stream():

    async def astream():

        llm = Text(text="What is the capital of France?")
        async for res in llm.astream_togetherai():
            assert isinstance(res, str)

    asyncio.run(astream())


def test_together_ai_structured_extraction():

    async def astructured_extraction():

        llm = Text(text="What is the capital of France?")

        class ExtractedData(BaseModel):
            name: Annotated[str, Field(description="Name of the city")]
            description: Annotated[str, Field(description="Description of the city")]

        res = await llm.astructured_extraction_togetherai(
            output_class=ExtractedData,
        )

        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


# def test_togetherai_astream_structured_extraction():

#     llm = Text()

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
