def test_instructor():
    import openai
    import instructor
    from pydantic import BaseModel


    client = instructor.from_openai(openai.OpenAI())


    class User(BaseModel):
        name: str
        age: int


    user_stream = client.chat.completions.create_partial(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=User,
    )

    for user in user_stream:
        print(user)