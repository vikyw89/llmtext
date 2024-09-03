import asyncio
from typing import AsyncGenerator, Type
from uuid import uuid4

from llmtext.types import (
    Checkpoint,
    Event,
    Message,
    RunnableTool,
    ToolOutput,
    IsFinalResponse,
)
from llmtext.llm import LLM
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import logging

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        chat_llm: LLM = LLM(),
        tool_selector_llm: LLM = LLM(),
        evaluator_llm: LLM = LLM(),
        messages: list[ChatCompletionMessageParam] = [],
        tools: list[Type[RunnableTool]] = [],
        max_steps: int = 1,
    ):
        self.chat_llm = chat_llm
        self.tool_selector_llm = tool_selector_llm
        self.evaluator_llm = evaluator_llm
        self.messages = messages
        self.tools = tools
        self.max_steps = max_steps

    async def astream_events(self) -> AsyncGenerator[Event, None]:
        logger.info("Agent starting...")
        steps = 0
        while True:
            steps += 1

            # checkpoint
            checkpoint = Event(
                id=str(uuid4()),
                step=steps,
                type="checkpoint",
                content=Checkpoint(messages=self.messages, type="checkpoint"),
            )
            logger.info(f"Checkpoint: {checkpoint}")
            yield checkpoint

            # break conditions
            if steps >= self.max_steps:
                logger.info("Max steps reached")
                break
            elif self._arun_evaluator_llm() is True:
                logger.info("Final response reached")
                break

            # extract tools
            tools_to_call = await self._arun_tool_selector_llm()

            for tool in tools_to_call:
                yield Event(
                    id=str(uuid4()),
                    step=steps,
                    type="tool_call",
                    content=tool.to_tool_call(),
                )

            # call tools
            tools_output = await self._acall_tools(tool_calls=tools_to_call)

            for tool_output in tools_output:
                if isinstance(tool_output, BaseException):
                    yield Event(
                        id=str(uuid4()),
                        step=steps,
                        type="evaluation",
                        content=ToolOutput(
                            type="tool_output",
                            description=str(tool_output),
                            output=str(tool_output),
                            params={},
                            name="",
                        ),
                    )
                    logger.error(f"Error calling tool: {tool_output}")
                    continue

                yield Event(
                    id=str(uuid4()),
                    step=steps,
                    type="tool_output",
                    content=tool_output,
                )

            # add tools to messages
            stream = self._astream_chat_llm()

            event_id = str(uuid4())
            final_message_content = ""
            async for chunk in stream:
                yield Event(
                    id=event_id,
                    step=steps,
                    type="message_stream",
                    content=Message(role="assistant", content=chunk),
                )

            yield Event(
                id=event_id,
                step=steps,
                type="message",
                content=Message(role="assistant", content=final_message_content),
            )

            # run next step

    async def _astream_chat_llm(self) -> AsyncGenerator[str, None]:
        stream = self.chat_llm.astream_response_from_messages(self.messages)

        final_message_content = ""
        async for chunk in stream:
            yield chunk
            final_message_content += chunk

        self.messages.append({"role": "assistant", "content": final_message_content})

    async def _arun_tool_selector_llm(self) -> list[RunnableTool]:
        from llmtext.utils_fns import tools_to_tool_selector

        tool_selector = tools_to_tool_selector(tools=self.tools)

        completion = await self.tool_selector_llm.astructured_extraction_from_messages(
            messages=self.messages, output_class=tool_selector
        )

        return completion.tool_calls

    async def _arun_evaluator_llm(self) -> bool:
        completion = await self.evaluator_llm.astructured_extraction_from_messages(
            messages=self.messages, output_class=IsFinalResponse
        )

        return completion.is_final_response

    async def _acall_tools(self, tool_calls: list[RunnableTool]) -> list[ToolOutput]:
        tasks = []

        for tool in tool_calls:
            tasks.append(tool.acall_and_return_tool_output())

        tools_output: list[ToolOutput | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        parsed_tool_output = []

        for tool_output in tools_output:
            if isinstance(tool_output, BaseException):
                logger.error(f"Error calling tool: {tool_output}")
                continue
            parsed_tool_output.append(tool_output)

        self.messages.append(
            {
                "role": "assistant",
                "content": "\n".join([o.output for o in parsed_tool_output]),
            }
        )

        return parsed_tool_output
