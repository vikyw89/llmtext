import os
from typing import Annotated
from pydantic import BaseModel, Field


def test_tool_schema():
    from llmtext.tools.base import Tool

    class ToolInput(BaseModel):
        test: Annotated[str, Field(description="test field")]

    class Tool2(BaseModel):
        test2: str

    async def test_tool(input: ToolInput, input2: Tool2) -> str:
        """always call this tool"""
        return "test"

    tool = Tool(afn=test_tool)

    print(tool.to_openai_schema())

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    res = client.chat.completions.create(
        messages=[{"role": "user", "content": "call a tool for me"}],
        tools=[tool.to_openai_schema()],
        model="gpt-3.5-turbo",
        tool_choice="required",
    )

    print(res)


# def test_agent_tools():
#     from llmtext.agents.openai_agent import OpenaiAgent

#     agent = OpenaiAgent()

#     class ToolInput(BaseModel):
#         """test tool input"""

#         test: Annotated[str, Field(description="test field")]

#     async def test_tool(input: ToolInput) -> str:
#         """always call this tool"""
#         return "test"

#     agent.atools = test_tool
#     agent.add_message({"role": "user", "content": "call a tool for me"})
#     import asyncio

#     asyncio.run(agent.arun_step())
