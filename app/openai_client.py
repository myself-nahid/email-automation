# --- START OF FILE openai_client.py ---
from typing import Optional
import openai
from openai import AsyncOpenAI
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

try:
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI AsyncClient: {e}", exc_info=True)
    async_client = None


async def get_embedding(text: str, model="text-embedding-ada-002") -> Optional[np.ndarray]:
    if not async_client:
        logger.error("OpenAI AsyncClient is not initialized.")
        return None
    if not text or not text.strip():
        return None

    try:
        response = await async_client.embeddings.create(input=[text], model=model)
        if response and response.data:
            return np.array(response.data[0].embedding, dtype=np.float32)
        return None
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return None


async def chat_completion(messages: list, model="gpt-4") -> Optional[str]:
    if not async_client:
        logger.error("OpenAI AsyncClient is not initialized.")
        return "Error: AI service is not available."

    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages
        )
        if response and response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        return "Error: Could not get a valid response from AI."
    except Exception as e:
        logger.error(f"Error during chat completion: {e}", exc_info=True)
        return "Error: An unexpected issue occurred with the AI service."