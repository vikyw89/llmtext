from abc import abstractmethod
import inspect
import json
from typing import Annotated, Any, Awaitable, Callable, Type, get_type_hints
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel, Field, create_model


# class Tool:
#     def __init__(self, afn: Callable[..., Awaitable[Any]]) -> None:
#         self.afn = afn
#         self.name = afn.__name__

#         # Get type hints for input and output
#         type_hints = get_type_hints(afn)
#         if "return" not in type_hints:
#             raise Exception("Function must have a 'return' type hint")

#         self.output = type_hints["return"]

#         if not afn.__doc__:
#             raise Exception("Tool needs description")
#         self.description: str = afn.__doc__

#         # Infer input type from the function's parameters
#         self.sig = inspect.signature(afn)
#         self.params = self.sig.parameters

#         self.openai_schema = None
#         self.pydantic_model = None

#     def to_openai_schema(self) -> ChatCompletionToolParam:
#         # Create the Pydantic model dynamically
#         DynamicModel = self.to_pydantic_model()

#         self.openai_schema = DynamicModel.model_json_schema()

#         return ChatCompletionToolParam(
#             function=FunctionDefinition(
#                 name=self.name,
#                 description=self.description,
#                 parameters=self.openai_schema,
#             ),
#             type="function",
#         )

#     def to_pydantic_model(self) -> Type[BaseModel]:
#         fields = {}

#         for name, param in self.params.items():
#             # Determine the field type and default value
#             field_type = param.annotation
#             default_value = param.default if param.default is not param.empty else ...
#             fields[name] = (field_type, default_value)

#         # Create the Pydantic model dynamically
#         DynamicModel = create_model(self.name, **fields)

#         self.pydantic_model = DynamicModel
#         return self.pydantic_model


class RunnableTool(BaseModel):
    @abstractmethod
    async def arun(self) -> str:
        pass

    def to_context(self) -> str:
        tool_name = self.__class__.__name__
        return json.dumps(
            {
                "tool_name": tool_name,
            }
        )

    async def aget_tool_output(self) -> str:
        tool_name = self.__class__.__name__
        return json.dumps(
            {
                "tool_name": tool_name,
                "tool_description": self.__doc__,
                "tool_params": self.model_dump_json(),
                "tool_output": await self.arun(),
            }
        )