# LLMText

![logo](./docs/7f5db8f9-3ebe-4f32-a1b6-a38a6e13f1f6.jpeg)

## Overview

This codebase provides a set of tools and workflows for interacting with language models (LLMs) in an asynchronous manner. It includes functionalities for generating text, streaming text, and structured extraction from messages and text inputs. The codebase is designed to be modular and extensible, allowing for easy integration of new tools and workflows.

## Features

- **Asynchronous Text Generation**: Generate text asynchronously from user messages.
- **Streaming Text Generation**: Stream text responses asynchronously.
- **Structured Extraction**: Extract structured data from text inputs using predefined classes.
- **Workflows**: Define and execute complex workflows involving multiple tools and messages.

## Installation

To install the necessary dependencies, you can use the following command:

```bash
pip install llmtext
```

Ensure you have the required environment variables set up by creating a `.env` file in the root directory with the necessary configurations.

## Usage

### Running Tests

To run the tests, use the following command:

```bash
pytest
```

### Example Usage

Here is an example of how to use the asynchronous text generation functionality:

```python
from llmtext.messages_fns import agenerate
from llmtext.data_types import Message

async def main():
    text = await agenerate(
        messages=[Message(role="user", content="what's the weather today ?")]
    )
    print(text)

import asyncio
asyncio.run(main())
```

### Agentic Workflow

Here is an example of how to use the agentic workflow functionality:

```python
from llmtext.data_types import Message
from llmtext.workflows_fns import astream_agentic_workflow
from llmtext.data_types import RunnableTool
from typing import Annotated
from pydantic import Field

class SearchInternetTool(RunnableTool):
    """Tool to search internet"""

    query: Annotated[str, Field(description="search query")]

    async def arun(self) -> str:
        return f"there's no result for: {self.query}"

async def main():
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
    async for chunk in stream:
        print(chunk)

import asyncio
asyncio.run(main())
```

### Available Tests

- **test_messages.py**: Tests for message-related functionalities.
- **test_workflows.py**: Tests for workflow-related functionalities.
- **test_text.py**: Tests for text-related functionalities.
- **test_tool.py**: Tests for tool-related functionalities.

## Code Structure

- **tests/**: Contains all the test files.
  - **test_workflow/**: Tests for workflow functionalities.
  - **tools/**: Tests for tool functionalities.
- **llmtext/**: Contains the main codebase.
  - **utils_fns/**: Utility functions for converting messages and tools.
  - **data_types/**: Data types used throughout the codebase.
  - **messages_fns/**: Message-related functions.
  - **texts_fns/**: Text-related functions.
  - **workflows_fns/**: Workflow-related functions.

## Contributing

Contributions are welcome! Please follow the standard Git workflow:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch to your fork.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or issues, please open an issue on GitHub.
