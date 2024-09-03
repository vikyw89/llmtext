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
        messages=[
            {
                "role": "system",
                "content": """You are a language translation expert. Given a text and a target language, your task is to provide the translated text accurately and efficiently, ensuring that the translation is accurate and maintains the original meaning effectively. The translations should be clear, convey the intended requests in English, and use formal language appropriate for professional settings, incorporating polite expressions such as 'please' where suitable. Additionally, ensure that informal expressions like '저기' are replaced with more formal alternatives like '주십시오' or '요청합니다', and avoid excessive repetition of phrases like '요청드립니다' for better readability.""",
            },
            {
                "role": "user",
                "content": """# target language
korean

# text
We can observe that the model has somehow learned how to perform the task by providing it with just one example (i.e., 1-shot). 
For more difficult tasks, we can experiment with increasing the demonstrations (e.g., 3-shot, 5-shot, 10-shot, etc.).""",
            },
        ]
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
