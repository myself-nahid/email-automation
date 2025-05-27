import os
import time
import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use the new embeddings method with retry mechanism
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
)
async def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",  # latest embedding model
            input=text
        )
        return response.data[0].embedding
    except openai.RateLimitError as e:
        print(f"Rate limit error: {e}")
        time.sleep(60)  # Wait before retry
        raise
    except openai.APIError as e:
        print(f"API error: {e}")
        if "insufficient_quota" in str(e):
            raise ValueError("OpenAI API quota exceeded. Please check your billing details.")
        raise

# The chat completion part with retry mechanism
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
)
async def chat_completion(messages: list, model="gpt-4"):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except openai.RateLimitError as e:
        print(f"Rate limit error: {e}")
        time.sleep(60)  # Wait before retry
        raise
    except openai.APIError as e:
        print(f"API error: {e}")
        if "insufficient_quota" in str(e):
            raise ValueError("OpenAI API quota exceeded. Please check your billing details.")
        raise