from pydantic import BaseModel
from llmtext.llm import LLM


async def test_llm_agenerate_response_from_text():
    llm = LLM()

    completion = await llm.agenerate_response_from_text(text="hello")
    print(completion)
    assert isinstance(completion, str)


async def test_llm_agenerate_response_from_messages():
    llm = LLM()

    completion = await llm.agenerate_response_from_messages(
        messages=[{"role": "user", "content": "hello"}]
    )
    print(completion)
    assert isinstance(completion, str)


async def test_llm_astream_response_from_text():
    llm = LLM()

    completion = llm.astream_response_from_text(text="hello")

    final_content = ""
    async for chunk in completion:
        print(chunk)
        assert isinstance(chunk, str)

        final_content += chunk

    print(final_content)


async def test_llm_astream_response_from_messages():
    llm = LLM()

    completion = llm.astream_response_from_messages(
        messages=[{"role": "user", "content": "hello"}]
    )

    final_content = ""
    async for chunk in completion:
        print(chunk)

        final_content += chunk

    print(final_content)
    assert isinstance(final_content, str)


async def test_llm_astructured_extraction_from_text():
    llm = LLM()

    class ImaginaryCountry(BaseModel):
        """
        ImaginaryCountry
        """

        city: str
        country: str

    completion = await llm.astructured_extraction_from_messages(
        messages=[{"role": "user", "content": "create an imaginary country"}],
        output_class=ImaginaryCountry,
    )

    print(completion)
    assert isinstance(completion, ImaginaryCountry)


async def test_llm_astructured_extraction_from_messages():
    llm = LLM()

    class ImaginaryCountry(BaseModel):
        """
        ImaginaryCountry
        """

        city: str
        country: str

    completion = await llm.astructured_extraction_from_text(
        text="create an imaginary country", output_class=ImaginaryCountry
    )
    print(completion)
    assert isinstance(completion, ImaginaryCountry)


async def test_llm_astream_structured_extraction_from_text():
    class ImaginaryCountry(BaseModel):
        """
        ImaginaryCountry
        """

        city: str
        country: str

    llm = LLM()
    completion = await llm.astream_structured_extraction_from_text(
        text="create an imaginary country", output_class=ImaginaryCountry
    )
    async for chunk in completion:
        print(chunk.model_dump())

        assert isinstance(chunk, ImaginaryCountry)


async def test_llm_astream_structured_extraction_from_messages():
    class ImaginaryCountry(BaseModel):
        """
        ImaginaryCountry
        """

        city: str
        country: str

    llm = LLM()
    completion = await llm.astream_structured_extraction_from_messages(
        messages=[{"role": "user", "content": "create an imaginary country"}],
        output_class=ImaginaryCountry,
    )

    async for chunk in completion:
        print(chunk.model_dump())
        assert isinstance(chunk, ImaginaryCountry)
