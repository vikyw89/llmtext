import asyncio
from csv import DictWriter
import os
from typing import Annotated, Awaitable, Callable
from openai import AsyncOpenAI
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

CLIENT = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


async def agenerate_prompt(
    example_inputs: list[str], example_outputs: list[str], client=CLIENT
) -> str:
    logger.debug(f"Generating prompt: {example_inputs} -> {example_outputs}")
    from llmtext.messages_fns import agenerate

    SYSTEM_PROMPT = """Let's work this out in a step by step way to be sure we have the right answer.
Your task is to create an LLM prompt that acts as a function generator. 
Given an example input and an example output, generate a clear and concise prompt that, when given to an LLM, will produce similar outputs for similar inputs. 
The prompt should describe the task, incorporate the example, provide instructions for handling similar inputs, and be both general enough for variations and specific enough for consistent results. 
Return only the generated prompt, without any additional explanation or formatting.
"""

    example_pairs = ""
    for input, output in zip(example_inputs, example_outputs):
        example_pairs += f"""# Example input
{input}
# Example Output
{output}

"""

    completion = await agenerate(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": example_pairs,
            },
        ],
        client=client,
        temperature=0.8,
    )

    return completion


async def arun_prompt(
    system_prompt: str, example_inputs: list[str], client=CLIENT
) -> list[str]:
    from llmtext.messages_fns import agenerate

    tasks = []
    for example_input in example_inputs:
        tasks.append(
            agenerate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example_input},
                ],
                client=client,
            )
        )

    results: list[str] = await asyncio.gather(*tasks)

    return results


async def agenerate_prompt_and_optimize(
    example_inputs: list[str],
    example_outputs: list[str],
    scoring_fn: Annotated[
        Callable[[list[str], list[str], list[str]], Awaitable[float]],
        "A function that scores prompts",
    ],
    parallel_count: Annotated[
        int,
        Field(
            description="Number of search exploration to be run in parallel for each iteration"
        ),
    ] = 1,
    client=CLIENT,
) -> str:
    logger.debug(f"""# Generating prompt and optimizing:
# Parallel Count
{parallel_count}
""")
    logger.debug(f"Generating prompt: {example_inputs[0]} -> {example_outputs[0]}")
    tasks = []
    for _ in range(parallel_count):
        tasks.append(
            agenerate_prompt(
                example_inputs=example_inputs,
                example_outputs=example_outputs,
                client=client,
            )
        )

    prompts = await asyncio.gather(*tasks)

    tasks = []
    for prompt in prompts:
        tasks.append(
            arun_prompt(
                system_prompt=prompt,
                example_inputs=example_inputs,
                client=client,
            )
        )

    outputs = await asyncio.gather(*tasks)

    logger.debug("Scoring prompts")

    # score prompts
    best_score = 0.0
    best_prompt = ""
    with open("prompts_scores.csv", mode="a+") as f:
        writer = DictWriter(
            f,
            fieldnames=[
                "prompt",
                "score",
            ],
        )
        for prompt, output in zip(prompts, outputs):
            score = await scoring_fn(example_inputs, example_outputs, output)
            logger.debug(f"""Prompt
{prompt}
# Score
{score}""")
            writer.writerow({"prompt": prompt, "score": score})
            if score >= best_score:
                best_score = score
                best_prompt = prompt

    return best_prompt
