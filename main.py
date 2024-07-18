import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from openai import OpenAI

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)


@app.get("/ask/")
async def ask_question(question: str = Query(..., min_length=3, max_length=500)):
    """
    Ask a question and get a response from the AI model.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = client.chat.completions.create(
        model="llama-3-sonar-large-32k-online",
        messages=messages,
    )

    reply = response.choices[0].message.content

    return {"response": reply}
