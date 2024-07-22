import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import AsyncGenerator, AsyncIterable, Type
from typing import TypeVar
import instructor

T = TypeVar("T", bound=BaseModel)


async def agenerate(
    text: str,
    client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    ),
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    **kwargs,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def astream_generate(
    text: str,
    client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    ),
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    **kwargs,
) -> AsyncGenerator[str, None]:
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        stream=True,
        **kwargs,
    )

    async for chunk in stream:
        yield chunk.choices[0].delta.content or ""


async def astructured_extraction(
    text: str,
    output_class: Type[T],
    client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    ),
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_retries: int = 3,
    temperature: float = 0.0,
    instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> T:
    structured_client = instructor.from_openai(client, mode=instructor_mode)

    response = await structured_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        response_model=output_class,
        max_retries=max_retries,
        temperature=temperature,
        **kwargs,
    )

    return response


async def astream_structured_extraction(
    text: str,
    output_class: type[T],
    client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    ),
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_retries: int = 3,
    temperature: float = 0.0,
    instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> AsyncGenerator[T, None]:
    structured_client = instructor.from_openai(client, mode=instructor_mode)

    stream: AsyncIterable[output_class] = (
        structured_client.chat.completions.create_partial(
            model=model,
            response_model=output_class,
            temperature=temperature,
            max_retries=max_retries,
            messages=[{"role": "user", "content": text}],
            stream=True,
            **kwargs,
        )
    )

    return stream
