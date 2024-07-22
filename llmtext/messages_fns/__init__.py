import os
from typing import AsyncGenerator, Type
from llmtext.data_types import Message
from instructor.client import T
from openai import AsyncOpenAI
import instructor
from llmtext.utils_fns import messages_to_openai_messages


async def agenerate(
    messages: list[Message],
    client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    ),
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    **kwargs,
) -> str:
    parsed_messages = messages_to_openai_messages(messages=messages)

    response = await client.chat.completions.create(
        messages=parsed_messages,
        model=model,
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def astream_generate(
    messages: list[Message],
    client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    ),
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    **kwargs,
) -> AsyncGenerator[str, None]:
    parsed_messages = messages_to_openai_messages(messages=messages)

    stream = await client.chat.completions.create(
        messages=parsed_messages, model=model, stream=True, **kwargs
    )

    async for chunk in stream:
        yield chunk.choices[0].delta.content or ""


async def astructured_extraction(
    messages: list[Message],
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

    parsed_messages = messages_to_openai_messages(messages=messages)

    response = await structured_client.chat.completions.create(
        messages=parsed_messages,
        model=model,
        response_model=output_class,
        max_retries=max_retries,
        temperature=temperature,
        **kwargs,
    )

    return response


async def astream_structured_extraction(
    messages: list[Message],
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

    parsed_messages = messages_to_openai_messages(messages=messages)

    structured_client = instructor.from_openai(client=client, mode=instructor_mode)

    stream: AsyncGenerator[output_class, None] = (
        structured_client.chat.completions.create_partial(
            model=model,
            response_model=output_class,
            temperature=temperature,
            max_retries=max_retries,
            messages=parsed_messages,
            stream=True,
            **kwargs,
        )
    )
    return stream
