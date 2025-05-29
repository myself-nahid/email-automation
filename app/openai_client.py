# --- START OF FILE openai_client.py ---
from typing import Optional
import openai # Keep this for error types if needed
from openai import AsyncOpenAI # Import the ASYNCHRONOUS client
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Ensure the OpenAI client is initialized.
# For OpenAI SDK v1.0.0 and later, use AsyncOpenAI for async operations
try:
    # Use AsyncOpenAI for an async client
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI AsyncClient: {e}", exc_info=True)
    async_client = None # Set to None if initialization fails


async def get_embedding(text: str, model="text-embedding-ada-002") -> Optional[np.ndarray]:
    if not async_client: # Check the async_client
        logger.error("OpenAI AsyncClient is not initialized. Cannot get embedding.")
        return None
    if not text or not text.strip():
        logger.warning("Attempted to get embedding for empty or whitespace-only text.")
        # OpenAI API will error on empty string for 'input'
        # Example error: openai.BadRequestError: Error code: 400 - {'error': {'message': "'' is not a valid list of strings because [] is not a list of strings.", 'type': 'invalid_request_error', 'param': 'input', 'code': None}}
        # So, it's better to return None or raise an error here.
        return None

    logger.debug(f"Requesting embedding for text (first 100 chars): {text[:100]}...")
    try:
        # Call the method on the async_client instance
        response = await async_client.embeddings.create(input=[text], model=model)

        if response and response.data and response.data[0] and response.data[0].embedding:
            embedding_vector = np.array(response.data[0].embedding, dtype=np.float32)
            logger.debug(f"Successfully got embedding. Shape: {embedding_vector.shape}")
            return embedding_vector
        else:
            logger.error(f"Failed to extract embedding from OpenAI response. Response: {response}")
            return None

    except openai.APIError as e: # Catch specific OpenAI errors
        logger.error(f"OpenAI API Error during embedding generation: {e.status_code} - {e.message}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
        return None

# --- END OF FILE openai_client.py ---

# In app/openai_client.py

# ... (async_client initialization from above) ...

async def chat_completion(messages: list, model="gpt-4") -> Optional[str]:
    if not async_client:
        logger.error("OpenAI AsyncClient is not initialized. Cannot get chat completion.")
        return "Error: AI service is not available." # Or raise an exception

    logger.debug(f"Requesting chat completion with {len(messages)} messages...")
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages
        )
        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            logger.debug("Successfully got chat completion.")
            return content.strip() if content else None
        else:
            logger.error(f"Failed to extract content from OpenAI chat response. Response: {response}")
            return "Error: Could not get a valid response from AI."

    except openai.APIError as e:
        logger.error(f"OpenAI API Error during chat completion: {e.status_code} - {e.message}", exc_info=True)
        # Provide a more user-friendly error message
        if e.status_code == 401: return "Error: AI authentication failed."
        if e.status_code == 429: return "Error: AI service is temporarily overloaded. Please try again later."
        return f"Error: AI service unavailable ({e.status_code})."
    except Exception as e:
        logger.error(f"Unexpected error during chat completion: {e}", exc_info=True)
        return "Error: An unexpected issue occurred with the AI service."