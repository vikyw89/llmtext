import asyncio
import json
import os
from typing import (
    Annotated,
    AsyncGenerator,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from llmtext.data_types import Event, Message
from llmtext.data_types import RunnableTool
import logging
from uuid import uuid4
import instructor
from llmtext import messages_fns
from llmtext.utils_fns import tools_to_tool_selector

logger = logging.getLogger(__name__)


async def aextract_tools(
    messages: list[Message],
    tools: list[type[RunnableTool]],
    tool_selector_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    ),
    tool_selector_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> list[RunnableTool]:
    tool_selector = tools_to_tool_selector(tools=tools)

    tool_selector = await messages_fns.astructured_extraction(
        messages=messages,
        client=tool_selector_client,
        model=tool_selector_model,
        output_class=tool_selector,
        instructor_mode=instructor_mode,
        **kwargs,
    )

    return tool_selector.tool_calls


async def acall_tools(
    tools: list[RunnableTool],
) -> list[str]:

    tasks = []
    for tool in tools:
        tasks.append(tool.arun())

    tools_output = await asyncio.gather(*tasks, return_exceptions=True)

    parsed_tools_output = []
    for tool_output in tools_output:
        if isinstance(tool_output, Exception):
            parsed_tools_output.append(str(tool_output))
        else:
            parsed_tools_output.append(tool_output)

    return parsed_tools_output


async def aevaluate_results(
    messages: list[Message],
    evaluator_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    ),
    evaluator_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> int:

    class QAEvaluation(BaseModel):
        """QA Evaluation Result"""

        score: Annotated[
            int, Field(description="0 if it doesn't answer original query")
        ] = 0

    evaluation = await messages_fns.astructured_extraction(
        messages=messages,
        client=evaluator_client,
        model=evaluator_model,
        output_class=QAEvaluation,
        instructor_mode=instructor_mode,
        **kwargs,
    )

    return evaluation.score


async def astream_agentic_workflow(
    messages: list[Message],
    tools: list[type[RunnableTool]] = [],
    chat_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    ),
    tool_selector_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    ),
    tool_selector_instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    evaluator_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    ),
    evaluator_instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    chat_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    tool_selector_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    evaluator_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_step=2,
    min_score: Annotated[int, Field(ge=0, le=5)] = 3,
    **kwargs,
) -> AsyncGenerator[Event, None]:

    step = 1
    score = 0

    while True:
        # extract tools
        tool_calls = await aextract_tools(
            messages=messages,
            tools=tools,
            tool_selector_client=tool_selector_client,
            tool_selector_model=tool_selector_model,
            instructor_mode=tool_selector_instructor_mode,
            **kwargs,
        )

        for tool in tool_calls:
            yield Event(
                step=step,
                type="tool_call",
                id=str(uuid4()),
                content=json.dumps(tool.get_tool_call(), ensure_ascii=False),
            )

        # call tool
        tool_call_results = await acall_tools(tools=tool_calls)

        # feed to chat model
        tool_outputs_message = Message(
            role="assistant",
            content="\n".join(tool_call_results),
        )

        yield Event(
            step=step,
            type="tool_output",
            id=str(uuid4()),
            content=tool_outputs_message.model_dump_json(),
        )

        messages.append(tool_outputs_message)

        # stream chat model
        stream = messages_fns.astream_generate(
            messages=messages,
            client=chat_client,
            model=chat_model,
            **kwargs,
        )

        stream_id = str(uuid4())
        final_content = ""
        async for chunk in stream:
            yield Event(
                step=step,
                type="message_stream",
                id=stream_id,
                content=chunk,
            )
            final_content += chunk

        yield Event(
            step=step,
            type="message",
            id=stream_id,
            content=final_content,
        )

        step += 1
        if step > max_step:
            break

        # evaluate score
        score = await aevaluate_results(
            messages=messages,
            evaluator_client=evaluator_client,
            evaluator_model=evaluator_model,
            instructor_mode=evaluator_instructor_mode,
            **kwargs,
        )

        yield Event(
            step=step,
            type="score",
            id=str(uuid4()),
            content=f"Score: {score}",
        )

        if score >= min_score:
            break

        # add results to messages
        messages.append(Message(role="assistant", content=final_content))
