from llmtext.utils.tracing import trace
import asyncio


def test_tracing():
    @trace(verbose=True)
    async def test(a: int, b: int) -> int:
        return a * b

    asyncio.run(test(2, 5))
