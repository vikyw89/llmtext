import asyncio
import json
import os
from typing import (
    AsyncGenerator,
)
from openai import AsyncOpenAI
from llmtext.data_types import (
    AnswerFeedback,
    Event,
    Message,
    RunnableTool,
)
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
        tasks.append(tool.aget_tool_output())

    tools_output = await asyncio.gather(*tasks, return_exceptions=True)

    parsed_tools_output = []
    for tool_output in tools_output:
        if isinstance(tool_output, Exception):
            parsed_tools_output.append(str(tool_output))
        else:
            parsed_tools_output.append(
                json.dumps(tool_output, ensure_ascii=False, default=str)
            )

    return parsed_tools_output


async def aevaluate_results(
    messages: list[Message],
    evaluator_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    ),
    evaluator_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    instructor_mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> AnswerFeedback:

    feedback = await messages_fns.astructured_extraction(
        messages=messages,
        client=evaluator_client,
        model=evaluator_model,
        output_class=AnswerFeedback,
        instructor_mode=instructor_mode,
        **kwargs,
    )

    return feedback


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
    **kwargs,
) -> AsyncGenerator[Event, None]:
    logger.info(f"Starting agentic workflow with {len(tools)} tools")
    logger.info(f"Initial messages: {messages}")

    step = 0

    # retrieve system message
    system_message = None
    for message in messages:
        if message.role == "system":
            system_message = message
            break

    # retrieve last message from user
    last_user_message = messages[-1]

    while step <= max_step:
        logger.debug(f"Starting step {step}")

        step += 1

        # extract tools
        tool_calls = await aextract_tools(
            messages=messages,
            tools=tools,
            tool_selector_client=tool_selector_client,
            tool_selector_model=tool_selector_model,
            instructor_mode=tool_selector_instructor_mode,
            **kwargs,
        )
        logger.debug(f"Tool calls: {tool_calls}")
        for tool in tool_calls:
            yield Event(
                step=step,
                type="tool_call",
                id=str(uuid4()),
                content=json.dumps(tool.get_tool_call(), ensure_ascii=False),
            )

        # call tool
        tool_call_results = await acall_tools(tools=tool_calls)
        logger.debug(f"Tool call results: {tool_call_results}")
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

        final_step_message = Message(role="assistant", content=final_content)
        logger.debug(f"Final step {step} message: {final_step_message}")

        # add results to messages
        messages.append(final_step_message)

        # won't do evaluation if next step is max
        if step + 1 > max_step:
            break

        # self reflect
        feedback = await aevaluate_results(
            messages=(
                [system_message]
                if system_message
                else [] + [last_user_message] + [final_step_message]
            ),
            evaluator_client=evaluator_client,
            evaluator_model=evaluator_model,
            instructor_mode=evaluator_instructor_mode,
            **kwargs,
        )

        if feedback.answer_feedback is None:
            break

        logger.debug(f"Feedback for step {step}: {feedback}")
        yield Event(
            step=step,
            type="feedback",
            id=str(uuid4()),
            content=feedback.answer_feedback,
        )

        messages.append(Message(role="user", content=feedback.answer_feedback))
        logger.debug(f"Final messages for step {step}: {messages}")

    logger.info(f"Ending agentic workflow within {step} steps")
    logger.info(f"Final response: {messages[-1].content}")
