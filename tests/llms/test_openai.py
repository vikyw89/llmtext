import json
from typing import Annotated
from pydantic import BaseModel, Field


def test_openai_arun():
    import asyncio

    async def arun():
        from llmtext.llms.openai import OpenAILLM

        llm = OpenAILLM()
        res = await llm.arun(text="What is the capital of France?")
        assert res is not None

    asyncio.run(arun())


def test_openai_stream():
    import asyncio

    async def astream():
        from llmtext.llms.openai import OpenAILLM

        llm = OpenAILLM()

        async for res in llm.astream("What is the capital of France?"):
            assert isinstance(res, str)

    asyncio.run(astream())


def test_openai_structured_extraction():
    import asyncio

    async def astructured_extraction():
        from llmtext.llms.openai import OpenAILLM

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


def test_openai_split_text():
    from llmtext.llms.openai import OpenAILLM

    llm = OpenAILLM()

    with open(file="./tests/llms/raw.txt", mode="r") as f:
        text = f.read()

    res = llm._chunk_text_by_line(text=text, max_tokens=500)

    with open("./tests/llms/output.json", "w+") as f:
        f.write(json.dumps(res, indent=4))

    with open(file="./tests/llms/output.md", mode="a+") as f:
        for i in res:
            f.write(i + "\n")
    assert res is not None
