import asyncio
import os
from typing import Annotated
from openai import AsyncOpenAI, BaseModel
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

CLIENT = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


async def agenerate_prompt(prompt: str, client=CLIENT) -> str:
    from llmtext.messages_fns import agenerate

    SYSTEM_PROMPT = """Here's a refined version of the LLM prompt:

# AI Prompt Engineering Specialist

You are an AI expert in crafting optimal prompts for language models, with a focus on clarity, effectiveness, and ethical considerations.

## Core Capabilities:
- Analyze user requirements for AI assistants
- Generate precise, tailored system prompts
- Optimize for specific roles, tasks, and target audiences

## Key Skills:
- Advanced natural language processing
- AI behavior and interaction modeling
- Task-specific prompt optimization
- Ethical AI design principles

## Traits:
- Analytical and meticulous
- Creative problem-solver
- Adaptable to diverse AI applications

## Ethical Framework:
- Prioritize safety and beneficial outcomes
- Avoid prompts that could lead to harmful or biased behavior
- Uphold user privacy and data protection standards

## Operational Protocol:
1. Analyze user's AI assistant requirements
2. Craft a system prompt incorporating:
   - Clear role definition
   - Concise function and expertise summary
   - Defined traits and communication style
   - Ethical guidelines and limitations
   - Task-specific focus
   - Direct, unambiguous language
   - Audience-appropriate content
3. Use markdown for enhanced readability
4. Explain design choices if requested

Respond with well-structured, effective prompts tailored to user specifications.
"""
    completion = await agenerate(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        client=client,
    )

    return completion.strip("```")


class Feedback(BaseModel):
    """Feedback to the answer"""

    feedback: Annotated[str, Field(description="Feedback to the answer")]
    score: Annotated[
        int, Field(description="Score of the answer, from 0 to 5, 0 bad 5 perfect")
    ]


async def arun_prompt(system_prompt: str, input: str, client=CLIENT) -> str:
    from llmtext.messages_fns import agenerate

    completion = await agenerate(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        client=client,
    )

    return completion


async def arun_feedback(prompt: str, input: str, client=CLIENT) -> Feedback:
    from llmtext.messages_fns import astructured_extraction

    feedback = await astructured_extraction(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        client=client,
        output_class=Feedback,
    )

    return feedback


async def agenerate_prompt_and_optimize(
    prompt: str,
    sequence_count: Annotated[
        int, Field(description="Number of sequential iterations")
    ],
    parallel_count: Annotated[
        int,
        Field(
            description="Number of search exploration to be run in parallel for each iteration"
        ),
    ],
    client=CLIENT,
) -> str:
    logger.debug(f"""# Generating prompt and optimizing:

# Prompt
{prompt}

# Sequence Count
{sequence_count}

# Parallel Count
{parallel_count}
""")
    from csv import DictWriter

    with open("prompts_and_score.csv", mode="w") as f:
        writer = DictWriter(f, fieldnames=["prompt", "score", "feedback"])
        writer.writeheader()

    depth = 0

    while True:
        logger.debug(f"Depth: {depth}")
        logger.debug(f"Prompt:\n{prompt}")
        depth += 1

        # break case
        if depth > sequence_count:
            break

        # generate prompt
        tasks = []
        for _ in range(parallel_count):
            tasks.append(agenerate_prompt(prompt, client=client))

        prompts: list[str] = await asyncio.gather(*tasks)

        # score the prompts
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                arun_feedback(
                    prompt=prompts[i],
                    input=prompt,
                    client=client,
                )
            )

        feedbacks: list[Feedback] = await asyncio.gather(*tasks)

        # select the best prompt
        best_prompt = ""
        best_score = 0
        best_feedback = Feedback(feedback="", score=0)

        for i in range(len(feedbacks)):
            with open("prompts_and_score.csv", mode="a+") as f:
                writer = DictWriter(f, fieldnames=["prompt", "score", "feedback"])
                writer.writerow(
                    {
                        "prompt": prompts[i],
                        "score": feedbacks[i].score,
                        "feedback": feedbacks[i].feedback,
                    }
                )
            if feedbacks[i].score > best_score:
                best_prompt = prompts[i]
                best_score = feedbacks[i].score
                best_feedback = feedbacks[i]

        prompt = f"""# Refine the following llm prompt, using this feedback {best_feedback.feedback}
        
{best_prompt}
"""
    return prompt
