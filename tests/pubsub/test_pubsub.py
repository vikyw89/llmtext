import pytest


@pytest.mark.asyncio
async def test_pubsub():
    from async_pubsub_py import PubSub

    pubsub = PubSub()

    async def subscriber_callback(message: str) -> None:
        print(f"Received message: {message}")

    # Subscribe to a topic
    pubsub.subscribe("topic1", subscriber_callback)

    # Publish a message to the topic
    await pubsub.publish("topic1", "Hello, World!")
