import asyncio
from typing import Annotated, AsyncIterable, Iterable, Literal, TypedDict, Union

from pydantic import BaseModel, Field
from llmtext.chat import Chat
from llmtext.tools import RunnableTool
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class Event(TypedDict):
    type: Literal["tool_call", "tool_output", "message_stream", "message"]
    id: str
    content: str


class Agent:
    def __init__(self, chat: Chat, tools: list[type[RunnableTool]], **kwargs) -> None:
        self.chat = chat
        self.tools = tools
        self.selected_tools: list[RunnableTool] = []
        self.tools_output: list[str] = []
        self.is_final = False

    async def arun_tool_selector(self, **kwargs) -> list[RunnableTool]:
        tuple_tools = tuple(self.tools)
        tools = list[Union[*tuple_tools]]  # type: ignore

        class ToolSelector(BaseModel):
            """Selected tools"""

            choices: Annotated[tools, Field(description="Selected tool to call")] = []  # type: ignore

        response = await self.chat.astructured_extraction(
            output_class=ToolSelector, **kwargs
        )

        self.selected_tools = response.choices

        parsed = []

        for tool in self.selected_tools:
            parsed.append(tool.to_context())

        del self.chat.messages[-1]
        if len(parsed) > 0:
            context = "\n".join(parsed)
            self.chat.messages.append(
                {"role": "assistant", "content": f"I will call these tools:\n{context}"}
            )

        return response.choices

    async def arun_tool(self, **kwargs) -> list[str]:
        tasks = []
        for tool in self.selected_tools:
            tasks.append(tool.aget_tool_output(**kwargs))

        self.tools_output = await asyncio.gather(*tasks)

        if len(self.tools_output) > 0:
            context = "Here's the result of the tool call: \n"

            for tool, output in zip(self.selected_tools, self.tools_output):
                context += f"""ToolOutput: 
{output}\n"""

        self.chat.messages.append({"role": "assistant", "content": context})
        return self.tools_output

    async def arun_synthesize(self, **kwargs):
        class Response(BaseModel):
            """Synthesized response"""

            response: Annotated[str, Field(description="Response to be sent to user")]
            is_final: Annotated[bool, Field(description="Is the response final?")] = (
                True
            )

        res = await self.chat.astructured_extraction(output_class=Response, **kwargs)
        del self.chat.messages[-1]
        self.chat.messages.append({"role": "assistant", "content": res.response})
        self.is_final = res.is_final

    async def astream_synthesize(self, **kwargs) -> AsyncIterable[str]:
        class Response(BaseModel):
            """Synthesized response"""

            response: Annotated[str, Field(description="Response to be sent to user")]
            is_final: Annotated[bool, Field(description="Is the response final?")] = (
                True
            )

        stream = await self.chat.astream_structured_extraction(
            output_class=Response, **kwargs
        )

        final_stream = ""
        async for chunk in stream:
            if chunk is None or chunk.response is None:
                continue
            new_token = chunk.response[len(final_stream) :]
            if len(new_token) == 0:
                continue
            yield new_token
            final_stream = chunk.response
            self.is_final = chunk.is_final

        del self.chat.messages[-1]
        self.chat.messages.append({"role": "assistant", "content": final_stream})

    async def arun_all(self, **kwargs) -> str:
        final_messages = []
        while self.is_final is False:
            logger.debug(self.chat.messages)
            await self.arun_tool_selector(**kwargs)
            logger.debug(self.chat.messages[-1])
            await self.arun_tool(**kwargs)
            logger.debug(self.chat.messages[-1])
            await self.arun_synthesize(**kwargs)
            logger.debug(self.chat.messages[-1])
            final_messages.append(self.chat.messages[-1].get("content", ""))

        return final_messages[-1]

    async def astream_events(self, **kwargs) -> AsyncIterable[Event]:
        while self.is_final is False:
            await self.arun_tool_selector(**kwargs)
            yield Event(type="tool_call", content="", id=str(uuid4()))
            await self.arun_tool(**kwargs)
            yield Event(type="tool_output", content="", id=str(uuid4()))
            stream = self.astream_synthesize(**kwargs)
            id = str(uuid4())
            async for chunk in stream:
                yield Event(type="message_stream", content=chunk, id=id)
            final_message = str(self.chat.messages[-1].get("content", ""))
            yield Event(type="message", content=final_message, id=id)
