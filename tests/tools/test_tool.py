from typing import Annotated, Any, Literal, Type, Union
from pydantic import BaseModel, Field
import pytest
from pydantic import BaseModel



@pytest.mark.asyncio
async def test_tool_class():
    from llmtext.text import Text

    class SearchNewsTool(BaseModel):
        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"

    class SearchInternetTool(BaseModel):
        query: Annotated[str, Field(description="search query")]

        async def arun(self) -> str:
            return f"there's no result for: {self.query}"


    tool_list = [SearchInternetTool, SearchNewsTool]
    tool_tuple = tuple(tool_list)
    tools = list[Union[*tool_tuple]]

    class ToolSelector(BaseModel):
        choice: Annotated[tools, Field(description="Selected tool")] = []
        response: Annotated[str, Field(description="Response to be sent to user")]

    text = Text(text="find me tesla news and search internet for tesla price")

    res = await text.astructured_extraction(output_class=ToolSelector)

    print(res)
