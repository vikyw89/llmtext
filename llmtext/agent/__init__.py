import asyncio
import json
from typing import Annotated, AsyncIterable, Literal, TypedDict, Union
from pydantic import BaseModel, Field
from llmtext.chat import Chat
from llmtext.tools import RunnableTool
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class Event(TypedDict):
    step: int
    type: Literal["tool_call", "tool_output", "message_stream", "message"]
    id: str
    content: str


class Agent:
    def __init__(
        self,
        chat: Chat,
        tools: list[type[RunnableTool]],
        max_step=20,
        min_score: Annotated[int, Field(ge=0, le=5)] = 3,
        prune_tools_between_steps=False,
    ) -> None:
        self.chat = chat
        self.tools = tools
        self.selected_tools: list[RunnableTool] = []
        self.tools_output: list[str] = []
        self.max_step = max_step
        self.step = 0
        self.min_score = min_score
        self.score = 0
        self.prune_tools_between_steps = prune_tools_between_steps

    async def arun_tool_selector(self, **kwargs) -> list[RunnableTool]:
        tuple_tools = tuple(self.tools)
        tools = list[Union[*tuple_tools]]  # type: ignore

        class ToolSelector(BaseModel):
            """Selected tools"""

            choices: Annotated[tools, Field(description="Selected tool to call")] = []  # type: ignore

        response = await self.chat.astructured_extraction(
            output_class=ToolSelector, **kwargs
        )
        del self.chat.messages[-1]

        self.selected_tools = response.choices

        tool_calls = []

        for tool in self.selected_tools:
            tool_calls.append(tool.to_context())

        if len(tool_calls) > 0:
            self.chat.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(tool_calls, ensure_ascii=False),
                }
            )
        return response.choices

    async def arun_tool(self, **kwargs) -> list[str]:
        tasks = []
        for tool in self.selected_tools:
            tasks.append(tool.aget_tool_output(**kwargs))

        self.tools_output = await asyncio.gather(*tasks)

        if len(self.tools_output) > 0:
            tools_call = []

            for tool, output in zip(self.selected_tools, self.tools_output):
                tools_call.append(output)

            self.chat.messages.pop(-1)

            self.chat.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        tools_call, ensure_ascii=False, default=lambda o: o.__dict__
                    ),
                }
            )

        return self.tools_output

    async def arun_synthesize(self, **kwargs) -> str:

        response = await self.chat.arun(**kwargs)

        self.chat.messages.append({"role": "assistant", "content": response})

        return response

    async def astream_synthesize(self, **kwargs) -> AsyncIterable[str]:

        stream = self.chat.astream(**kwargs)

        final_stream = ""
        async for chunk in stream:
            yield chunk
            final_stream += chunk

        self.chat.messages.append({"role": "assistant", "content": final_stream})

    async def arun_score(self, **kwargs) -> int:
        class ResponseScore(BaseModel):
            """Response score"""

            score: Annotated[
                int, Field(description="Response score, from 0 - bad to 5 - perfect")
            ]

        response = await self.chat.astructured_extraction(
            output_class=ResponseScore, **kwargs
        )
        self.chat.messages.pop(-1)

        return response.score

    async def arun_all(self, **kwargs) -> str:
        final_response = ""
        while self.score <= self.min_score and self.step <= self.max_step:
            await self.arun_tool_selector(**kwargs)
            await self.arun_tool(**kwargs)
            final_response = await self.arun_synthesize(**kwargs)

            self.score = await self.arun_score(**kwargs)
        return final_response

    async def astream_events(self, **kwargs) -> AsyncIterable[Event]:
        while self.score <= self.min_score and self.step <= self.max_step:
            self.step += 1

            await self.arun_tool_selector(**kwargs)
            last_message = self.chat.messages[-1].get("content", "")
            if isinstance(last_message, str):
                yield Event(
                    step=self.step,
                    type="tool_call",
                    content=last_message,
                    id=str(uuid4()),
                )

            await self.arun_tool(**kwargs)
            last_message = self.chat.messages[-1].get("content", "")
            if isinstance(last_message, str):
                yield Event(
                    step=self.step,
                    type="tool_output",
                    content=last_message,
                    id=str(uuid4()),
                )

            stream = self.astream_synthesize(**kwargs)
            id = str(uuid4())
            async for chunk in stream:
                yield Event(step=self.step, type="message_stream", content=chunk, id=id)

            final_message = str(self.chat.messages[-1].get("content", ""))

            yield Event(step=self.step, type="message", content=final_message, id=id)

            if self.prune_tools_between_steps:
                self.prune_tool_messages()

            self.score = await self.arun_score(**kwargs)

    def prune_tool_messages(self):
        if len(self.tools_output) > 0:
            self.chat.messages.pop(-2)
            self.tools_output = []
            self.selected_tools = []
