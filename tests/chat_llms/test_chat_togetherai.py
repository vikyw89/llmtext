from typing import Annotated
from pydantic import BaseModel, Field
from llmtext.chat_llms.togetherai import ChatTogetherAI
import asyncio


def test_togetherai_arun():

    async def arun():

        llm = ChatTogetherAI()
        llm.add_message({"role": "system", "content": "You are a helpful assistant."})
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        res = await llm.arun()
        assert res is not None

    asyncio.run(arun())


def test_togetherai_stream():

    async def astream():

        llm = ChatTogetherAI()
        llm.add_message({"role": "system", "content": "You are a helpful assistant."})
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        stream = llm.astream()

        async for chunk in stream:
            assert isinstance(chunk, str)

    asyncio.run(astream())


def test_togetherai_astructured_extraction():

    async def astructured_extraction():
        from llmtext.chat_llms.togetherai import ChatTogetherAI

        llm = ChatTogetherAI()

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


# def test_chat_togetherai_astream_structured_extraction():

#     llm = ChatTogetherAI()

#     class ExtractedData(BaseModel):
#         name: Annotated[str, Field(description="Name of the city")]
#         description: Annotated[str, Field(description="Description of the city")]

#     async def arun():
#         llm.messages = [
#             {
#                 "role": "system",
#                 "content": "Extract what the user asks from the following conversations.",
#             },
#             {
#                 "role": "user",
#                 "content": "The capital of Germany is Berlin. It's a beautiful city. The capital of France is Paris. It's a beautiful city.",
#             },
#         ]

#         res = await llm.astream_structured_extraction(
#             output_class=ExtractedData,
#         )

#         assert isinstance(res, AsyncGenerator)

#         async for chunk in res:
#             assert isinstance(chunk, ExtractedData)
#             print(chunk.model_dump())

#     asyncio.run(arun())
