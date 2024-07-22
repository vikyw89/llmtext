from typing import Annotated
from pydantic import Field
import pytest

from llmtext.data_types import Message, RunnableTool


@pytest.mark.asyncio
async def test_agenerate():
    from llmtext.messages_fns import agenerate

    text = await agenerate(
        messages=[Message(role="user", content="what's the weather today ?")]
    )

    print(text)
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_astream_generate():
    from llmtext.messages_fns import astream_generate

    stream = astream_generate(
        messages=[Message(role="user", content="what's the weather today ?")]
    )

    async for text in stream:
        print(text)
        assert isinstance(text, str)


@pytest.mark.asyncio
async def test_astructured_extraction():
    from llmtext.messages_fns import astream_structured_extraction

    class SearchInternetTool(RunnableTool):
        """Tool to search internet"""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    stream = await astream_structured_extraction(
        messages=[Message(role="user", content="what's the weather today ?")],
        output_class=SearchInternetTool,
    )

    async for tool in stream:
        print(tool)
        assert isinstance(tool, SearchInternetTool)
