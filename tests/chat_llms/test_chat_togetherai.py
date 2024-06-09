from typing import Annotated
from pydantic import BaseModel, Field
from llmtext.chat_llms.base import ChatCompletionChunk


def test_togetherai_arun():
    import asyncio

    async def arun():
        from llmtext.chat_llms.togetherai import ChatTogetherAI

        llm = ChatTogetherAI()
        llm.add_message({"role": "system", "content": "You are a helpful assistant."})
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        res = await llm.arun()
        assert res is not None

    asyncio.run(arun())


def test_togetherai_stream():
    import asyncio

    async def astream():
        from llmtext.chat_llms.togetherai import ChatTogetherAI

        llm = ChatTogetherAI()
        llm.add_message({"role": "system", "content": "You are a helpful assistant."})
        llm.add_message({"role": "user", "content": "What is the capital of France?"})
        stream = await llm.astream()

        async for chunk in stream:
            assert isinstance(chunk, ChatCompletionChunk)

    asyncio.run(astream())


def test_togetherai_structured_extraction():
    import asyncio

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
