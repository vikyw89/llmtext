import os
from typing import Annotated
from pydantic import Field
import pytest
import logging
from openai import AsyncOpenAI


@pytest.mark.asyncio
async def test_arun_tool_selector(caplog):
    caplog.set_level(logging.DEBUG)
    from llmtext.agent import Agent

    from llmtext.chat import Chat
    from llmtext.tools import RunnableTool

    class SearchTool(RunnableTool):
        """Use this tool to search for news articles."""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}, please try again"

    class CalculateTool(RunnableTool):
        """Use this tool to calculate the sum of two numbers."""

        a: int
        b: int

        async def arun(self) -> int:
            return self.a + self.b

    class MultiplyTool(RunnableTool):
        """Use this tool to multiply 2 numbers"""

        a: int
        b: int

        async def arun(self) -> int:
            return self.a + self.b

    agent = Agent(
        chat=Chat(
            messages=[
                {
                    "role": "user",
                    "content": "what's 1000 + 2000 + 3000 + 4000 ?",
                }
            ],
            openai_client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            ),
            openai_model="anthropic/claude-3-haiku",
        ),
        tools=[SearchTool, CalculateTool, MultiplyTool],
    )

    await agent.arun_all()

    for message in agent.chat.messages:
        print(message.get("content"))


@pytest.mark.asyncio
async def test_stream_synthesize(caplog):
    caplog.set_level(logging.DEBUG)
    from llmtext.agent import Agent
    from llmtext.chat import Chat
    from llmtext.tools import RunnableTool

    class SearchTool(RunnableTool):
        """Use this tool to search for news articles."""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}, please try again"

    class CalculateTool(RunnableTool):
        """Use this tool to calculate the sum of two numbers."""

        a: int
        b: int

        async def arun(self) -> int:
            return self.a + self.b

    class MultiplyTool(RunnableTool):
        """Use this tool to multiply 2 numbers"""

        a: int
        b: int

        async def arun(self) -> int:
            return self.a + self.b

    agent = Agent(
        chat=Chat(
            messages=[
                {
                    "role": "user",
                    "content": "what's 1000 + 2000 + 3000 + 4000 ?",
                }
            ],
            openai_client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            ),
            openai_model="anthropic/claude-3-haiku",
        ),
        tools=[SearchTool, CalculateTool, MultiplyTool],
    )

    stream = agent.astream_synthesize()

    async for chunk in stream:
        print(chunk)


@pytest.mark.asyncio
async def test_stream_events_haiku(caplog):
    caplog.set_level(logging.DEBUG)
    from llmtext.agent import Agent
    from llmtext.chat import Chat
    from llmtext.tools import RunnableTool

    class SearchTool(RunnableTool):
        """Use this tool to search for news articles."""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}, please try again"

    class CalculateTool(RunnableTool):
        """Use this tool to calculate the sum of two numbers."""

        a: int
        b: int

        async def arun(self) -> int:
            return self.a + self.b

    class MultiplyTool(RunnableTool):
        """Use this tool to multiply 2 numbers"""

        a: int
        b: int

        async def arun(self) -> int:
            return self.a * self.b

    agent = Agent(
        chat=Chat(
            messages=[
                {
                    "role": "system",
                    "content": "Let's think step by step. Always response in markdown format.",
                },
                {
                    "role": "user",
                    "content": "Write a comprehensive comparison between NVIDIA vs Intel. List all factors, give stars between 0 to 5 and sources. Tally the sources score at the end",
                },
            ],
            openai_client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            ),
            openai_model="anthropic/claude-3-haiku",
        ),
        tools=[SearchTool, CalculateTool, MultiplyTool],
    )

    stream = agent.astream_events()

    async for chunk in stream:
        print(chunk)
        print("\n")
