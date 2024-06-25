from typing import Annotated, AsyncGenerator
from pydantic import BaseModel, Field
from llmtext.chat_llms.togetherai import ChatTogetherAI
import asyncio


def test_togetherai_arun():

    async def arun():

        llm = ChatTogetherAI()
        res = await llm.arun(messages=[{"role": "user", "content": "heloooooo"}])
        assert res is not None

    asyncio.run(arun())


def test_togetherai_stream():

    async def astream():

        llm = ChatTogetherAI()
        stream = llm.astream(messages=[{"role": "user", "content": "helooooo"}])

        async for chunk in stream:
            assert isinstance(chunk, str)

    asyncio.run(astream())


def test_togetherai_astructured_extraction():

    async def astructured_extraction():
        from llmtext.chat_llms.togetherai import ChatTogetherAI

        llm = ChatTogetherAI()

        class ExtractedData(BaseModel):
            questions: Annotated[
                list[str], Field(description="The questions asked by the user")
            ]

        res = await llm.astructured_extraction(
            output_class=ExtractedData,
            messages=[
                {
                    "role": "system",
                    "content": "Extract what the user asks from the following conversations.",
                },
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
            ],
        )
        print("res", res)
        assert isinstance(res, ExtractedData)

    asyncio.run(astructured_extraction())


# def test_chat_togetherai_astream_structured_extraction():

#     llm = ChatTogetherAI()

#     class ExtractedData(BaseModel):
#         name: Annotated[str, Field(description="Name of the city")]
#         description: Annotated[str, Field(description="Description of the city")]

#     async def arun():

#         res = await llm.astream_structured_extraction(
#             output_class=ExtractedData,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "Extract what the user asks from the following conversations.",
#                 },
#                 {
#                     "role": "user",
#                     "content": "The capital of Germany is Berlin. It's a beautiful city. The capital of France is Paris. It's a beautiful city.",
#                 },
#             ],
#         )

#         assert isinstance(res, AsyncGenerator)

#         async for chunk in res:
#             assert isinstance(chunk, ExtractedData)
#             print(chunk.model_dump())

#     asyncio.run(arun())
