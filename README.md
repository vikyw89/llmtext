# llmtext

![alt text](/docs/7f5db8f9-3ebe-4f32-a1b6-a38a6e13f1f6.jpeg)

`llmtext` is a simple yet powerful library designed to interact with large language models (LLMs) as straightforward functions. It provides easy-to-use interfaces for both input-output text transformations and input-to-Pydantic class conversions, leveraging the power of open-source LLMs and OpenAI's schema.

## Features

- **Input Text, Output Text**: Seamlessly generate text outputs from text inputs using large language models.
- **Input Text, Output Pydantic Class**: Convert text inputs directly into structured Pydantic classes for better data validation and manipulation.
- **OpenAI Schema Support**: Utilize OpenAI's schema for consistent and robust text processing.
- **OpenSource LLMs through TogetherAI**: Access a variety of open-source LLMs via TogetherAI for flexible and cost-effective solutions.
- **Async by default**: Asynchronous by default
- **Robusts**: configured with retry and self healing loop for structured extraction

## Installation

You can install `llmtext` via pip:

```bash
pip install llmtext
```

## Usage

### Text to Text Transformation

To generate text outputs from text inputs:

```python
from llmtext.text import Text

text = Text(text="What is the capital of France ?")
res = await llm.arun(text="What is the capital of France?")
```

### Text to Pydantic Class Transformation

To convert text inputs into a Pydantic class:

```python
from llmtext.text import Text

text = Text(text="The city of France is Paris. It's a beautiful city.")

class ExtractedData(BaseModel):
    name: Annotated[str, Field(description="Name of the city")]
    description: Annotated[str, Field(description="Description of the city")]

res = await text.astructured_extraction(output_class=ExtractedData)
assert isinstance(res, ExtractedData)
```

### Text to Streaming Pydantic Class Transformation

To convert text inputs into a Pydantic class:

```python
from llmtext.text import Text

text = Text(text="The city of France is Paris. It's a beautiful city.")

class ExtractedData(BaseModel):
    name: Annotated[str, Field(description="Name of the city")]
    description: Annotated[str, Field(description="Description of the city")]

stream = await text.astream_structured_extraction(output_class=ExtractedData)

async for res in stream:
    assert isinstance(res, ExtractedData)
```

## Configuration

To configure `llmtext` for using OpenAI's schema or TogetherAI's open-source LLMs, you can set the necessary API keys in your environment variables or configuration file.

### Example Configuration to use togetherai, or openrouter or any other open-source LLMs that support openai schema

```.env
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.together.xyz/v1
OPENAI_MODEL=
```

or input it upon class initialization

```
llm = Text(
    text="What is the capital of France ?",
    openai_client=AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    ),
    openai_model="anthropic/claude-3-haiku",
)
```

## Contributing

We welcome contributions to `llmtext`. Please fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Special thanks to OpenAI for providing robust schema support and TogetherAI for enabling access to open-source LLMs.

---

For more information, please refer to the [documentation](https://github.com/vikyw89/llmtext).
