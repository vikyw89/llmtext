from typing import Annotated, Union
from pydantic import BaseModel, Field
import pytest


@pytest.mark.asyncio
async def test_tool_class():
    from llmtext.texts_fns import astructured_extraction

    class SearchNewsTool(BaseModel):
        """Tool to search news"""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    class SearchInternetTool(BaseModel):
        """Tool to search internet"""

        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    tool_list = [SearchInternetTool, SearchNewsTool]
    tool_tuple = tuple(tool_list)
    tools = list[Union[*tool_tuple]]  # type: ignore

    class ToolSelector(BaseModel):
        choice: Annotated[tools, Field(description="Selected tool")] = []  # type: ignore
        response: Annotated[str, Field(description="Response to be sent to user")]

    res = await astructured_extraction(text="Find a news", output_class=ToolSelector)

    print(res)
