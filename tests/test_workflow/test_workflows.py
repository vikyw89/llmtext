from typing import Annotated
from pydantic import Field
import pytest

from llmtext.data_types import RunnableTool


@pytest.mark.asyncio
async def test_aextract_tools():
    from llmtext.data_types import RunnableTool, Message
    from llmtext.workflows_fns import aextract_tools

    class SearchInternetTool(RunnableTool):
        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    tools = await aextract_tools(
        messages=[Message(role="user", content="what's the weather today ?")],
        tools=[SearchInternetTool],
    )

    assert len(tools) == 1

    for tool in tools:
        assert isinstance(tool, SearchInternetTool)


@pytest.mark.asyncio
async def test_acall_tools():
    from llmtext.data_types import RunnableTool
    from llmtext.workflows_fns import acall_tools

    class SearchInternetTool(RunnableTool):
        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    tool_output = await acall_tools(
        tools=[SearchInternetTool(query="what's the weather today ?")],
    )

    for output in tool_output:
        assert isinstance(output, str)


@pytest.mark.asyncio
async def test_aevaluate_result():
    from llmtext.data_types import Message
    from llmtext.workflows_fns import aevaluate_results

    score = await aevaluate_results(
        messages=[
            Message(role="user", content="what's the weather today ?"),
            Message(
                role="assistant",
                content="there's no result for: what's the weather today ?",
            ),
        ],
    )

    assert isinstance(score, int)

    print(score)


@pytest.mark.asyncio
async def test_astream_agentic_workflow():
    from llmtext.data_types import Message
    from llmtext.workflows_fns import astream_agentic_workflow

    class SearchInternetTool(RunnableTool):
        """Tool to search internet"""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    stream = astream_agentic_workflow(
        messages=[
            Message(role="user", content="what's the weather today ?"),
            Message(
                role="assistant",
                content="there's no result for: what's the weather today ?",
            ),
        ],
        tools=[SearchInternetTool],
    )

    ids = {}
    async for chunk in stream:
        print(chunk)
        if chunk["id"] not in ids:
            ids[chunk["id"]] = True
            print("\n")
