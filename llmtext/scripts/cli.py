import os
import typer


app = typer.Typer()


@app.callback()
def callback():
    """
    LLMText CLI
    """


@app.command()
def llm_run(text: str):
    """
    Call llm and receive a text
    """
    from llmtext.llms.openai import OpenAILLM
    import asyncio

    llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY", ""))

    res = asyncio.run(llm.arun(text=text))
    typer.echo(res)


@app.command()
def set_openai_api_key(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    typer.echo("API key set")


@app.command()
def set_together_ai_api_key(api_key: str):
    os.environ["TOGETHERAI_API_KEY"] = api_key
    typer.echo("API key set")
