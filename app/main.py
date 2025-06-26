# --- START OF FILE main.py ---
from app.openai_client import get_embedding, chat_completion
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from app.models import EmailInput, QuestionRequest, ChatMessage, EmailStatusUpdate
from app.email_processor import process_email, ProcessEmailError
from app.faiss_index import faiss_index
from app.db import (
    database, upsert_email, get_email_by_id, get_emails_by_ids,
    migrate_database, get_emails_in_date_range, update_email_status, metadata, engine
)
from app.chat_manager import add_message, get_chat_history
from datetime import datetime, timedelta, timezone
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = ["http://localhost",
           "http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# In-memory cache for conversational context. This gives the assistant "short-term memory".
# In a production environment with multiple server workers, this should be replaced with Redis.
CONVERSATION_CONTEXT_CACHE: Dict[str, List[str]] = {}

# --- FastAPI Lifecycle Events ---


@app.on_event("startup")
async def startup():
    await database.connect()
    metadata.create_all(engine)
    await migrate_database()
    try:
        faiss_index.load()
    except Exception as e:
        logger.error(
            f"Failed to load FAISS index on startup: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    try:
        faiss_index.save()
    except Exception as e:
        logger.error(
            f"Failed to save FAISS index on shutdown: {e}", exc_info=True)

# --- Helper Functions ---


def enhanced_build_smart_context(emails: List[dict], max_total_tokens=4000) -> str:
    """Creates a well-formatted string of email data for the LLM to analyze."""
    if not emails:
        return "No relevant emails found for the query."
    context_parts = []
    for i, email in enumerate(emails, 1):
        received_at_dt = email.get('received_at')
        date_str = received_at_dt.strftime(
            '%Y-%m-%d %H:%M UTC') if isinstance(received_at_dt, datetime) else "Unknown Date"
        status = "Unread" if email.get('is_unread') else "Read"

        part = f"""--- EMAIL {i} ---
Email ID: {email.get('email_id')}
Date: {date_str}
From: {email.get('from_', 'Unknown')}
Subject: {email.get('subject', 'No Subject')}
Status: {status}
Category: {email.get('category', 'general').capitalize()}
Content Snippet: {email.get('body', '')[:1000]}
"""
        if (sum(len(p) for p in context_parts) + len(part)) // 4 > max_total_tokens:
            break
        context_parts.append(part)
    return "\n".join(context_parts)


async def _retrieve_context_emails(account_id: str, question_lower: str) -> List[Dict]:
    """
    Applies multiple RAG strategies to find the most relevant emails for a query.
    """
    emails_for_context = []
    retrieved_ids = set()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=90)
    is_unread_query = "unread" in question_lower

    if is_unread_query:
        logger.info(
            f"Strategy 0: High-precision query for unread emails for account '{account_id}'.")
        db_emails = await get_emails_in_date_range(start_date, end_date, account_id=account_id, is_unread=True, limit=50)
        for row in db_emails:
            if row['email_id'] not in retrieved_ids:
                emails_for_context.append(dict(row))
                retrieved_ids.add(row['email_id'])

    category_to_check = None
    if any(k in question_lower for k in ['urgent', 'important']):
        category_to_check = 'urgent'
    elif any(k in question_lower for k in ['meeting', 'event', 'call']):
        category_to_check = 'meeting/event'
    if category_to_check:
        logger.info(
            f"Strategy 1: Querying for category '{category_to_check}' for account '{account_id}'.")
        db_emails = await get_emails_in_date_range(start_date, end_date, account_id=account_id, category=category_to_check, limit=50)
        for row in db_emails:
            if row['email_id'] not in retrieved_ids:
                emails_for_context.append(dict(row))
                retrieved_ids.add(row['email_id'])

    if not is_unread_query or len(emails_for_context) < 5:
        logger.info(
            f"Strategy 2: Performing semantic search with FAISS for account '{account_id}'.")
        try:
            q_embedding_data = await process_email({'id': 'query', 'account_id': account_id, 'subject': question_lower, 'body': ''}, generate_embedding_only=True)
            if q_embedding_data and q_embedding_data.get('embedding'):
                embedding_array = np.frombuffer(
                    q_embedding_data['embedding'], dtype=np.float32)
                distances, top_faiss_ids = faiss_index.search_with_scores(
                    embedding_array, top_k=50)
                if top_faiss_ids:
                    faiss_results = await get_emails_by_ids(top_faiss_ids, account_id=account_id)
                    for row in faiss_results:
                        if row['email_id'] not in retrieved_ids:
                            emails_for_context.append(dict(row))
                            retrieved_ids.add(row['email_id'])
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)

    if not emails_for_context:
        logger.info(
            f"Strategy 3: No results yet, falling back to general recent emails for account '{account_id}'.")
        db_emails = await get_emails_in_date_range(start_date, end_date, account_id=account_id, limit=20)
        for row in db_emails:
            if row['email_id'] not in retrieved_ids:
                emails_for_context.append(dict(row))
                retrieved_ids.add(row['email_id'])

    emails_for_context.sort(key=lambda x: x.get(
        'received_at', datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return emails_for_context

# --- Main API Endpoint ---


@app.post("/query/working")
async def working_query(request: QuestionRequest):
    try:
        account_id = request.account_id
        if not account_id:
            raise HTTPException(
                status_code=400, detail="Account ID is required and was not provided.")

        question_lower = request.question.lower()
        conversation_id = request.conversation_id

        drafting_keywords = ['draft', 'reply', 'write', 'respond', 'compose']
        is_draft_request = any(
            keyword in question_lower for keyword in drafting_keywords)

        # --- BRANCH 1: DRAFTING AGENT ---
        if is_draft_request:
            logger.info(
                f"Intent detected: DRAFTING for account '{account_id}'")
            target_email_id = request.email_id

            if not target_email_id and conversation_id:
                last_context_ids = CONVERSATION_CONTEXT_CACHE.get(
                    conversation_id)
                if last_context_ids:
                    target_email_id = last_context_ids[0]
                    logger.info(
                        f"Found target email ID '{target_email_id}' from conversation cache for follow-up.")

            if not target_email_id:
                return {"answer": "I'm not sure which email you want to reply to. Please ask about an email first, and then I can draft a reply."}

            target_email_dict = await get_email_by_id(target_email_id, account_id=account_id)
            if not target_email_dict:
                return {"answer": f"I'm sorry, I couldn't find email with ID '{target_email_id}' for your account."}

            target_email = dict(target_email_dict)
            drafting_system_prompt = f"""You are an AI assistant helping a user draft an email reply. The user is replying to an email from '{target_email.get('from_', 'unknown sender')}' with the subject 'Re: {target_email.get('subject', '')}'. Your task is to compose a clear, professional reply based on the user's instructions and the original email's content. Do NOT include a greeting or a closing. Focus ONLY on the body of the reply.

**ORIGINAL EMAIL CONTENT:**
---
{target_email.get('body', 'No Content')}
---
"""
            messages = []
            if conversation_id:
                history = await get_chat_history(conversation_id)
                messages.extend(
                    [{"role": msg['role'], "content": msg['content']} for msg in history])

            messages.extend([
                {"role": "system", "content": drafting_system_prompt},
                {"role": "user", "content": request.question}
            ])

            drafted_reply = await chat_completion(messages)

            if conversation_id and drafted_reply:
                await add_message(conversation_id, "user", request.question)
                assistant_response = f"Here is a draft reply:\n\n---\n\n{drafted_reply}"
                await add_message(conversation_id, "assistant", assistant_response)

            return {"answer": drafted_reply}

        # --- BRANCH 2: QUERY & ANALYSIS AGENT ---
        else:
            logger.info(
                f"Intent detected: QUERY/ANALYSIS for account '{account_id}'")

            emails_for_context = await _retrieve_context_emails(account_id, question_lower)

            if not emails_for_context:
                if conversation_id in CONVERSATION_CONTEXT_CACHE:
                    del CONVERSATION_CONTEXT_CACHE[conversation_id]
                return {"answer": "I couldn't find any relevant emails for your query."}

            if conversation_id:
                CONVERSATION_CONTEXT_CACHE[conversation_id] = [
                    e['email_id'] for e in emails_for_context]
                logger.info(
                    f"Cached {len(emails_for_context)} email IDs for conversation '{conversation_id}'.")

            context_text = enhanced_build_smart_context(emails_for_context)

            analysis_system_prompt = f"""You are a highly intelligent AI Email Assistant. Your primary goal is to answer questions by analyzing and summarizing the email data provided.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze, Don't Act:** Your role is to analyze and summarize. Do NOT offer to perform actions like sending emails yourself.
2.  **Cite Your Source:** When you refer to a specific email in your answer, you **MUST include its identifier** in this exact format: `(Email ID: <the_id>)`. This is mandatory so the user can issue follow-up commands.
3.  **Adhere to Data:** Base your answer *exclusively* on the content provided. If the information isn't there, state that clearly.
"""
            user_prompt_with_context = f"Analyze these emails and answer my question.\n\n**EMAIL DATA:**\n{context_text}\n\n**USER QUESTION:**\n{request.question}"

            messages = []
            if conversation_id:
                history = await get_chat_history(conversation_id)
                messages.extend(
                    [{"role": msg['role'], "content": msg['content']} for msg in history])

            messages.extend([
                {"role": "system", "content": analysis_system_prompt},
                {"role": "user", "content": user_prompt_with_context}
            ])

            answer = await chat_completion(messages)
            if not answer:
                answer = "I apologize, but I was unable to generate a response."

            if conversation_id:
                await add_message(conversation_id, "user", request.question)
                await add_message(conversation_id, "assistant", answer)

            return {"answer": answer}

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing your request.")

# --- Other API Endpoints ---


@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Email Assistant API"}


@app.post("/emails/batch_add")
async def batch_add_emails(emails_input: List[EmailInput]):
    logger.info(f"Received {len(emails_input)} emails for batch processing.")
    if not emails_input:
        return {"processed": 0, "failed": 0, "skipped": 0, "message": "No emails provided."}
    processed_count, failed_count, skipped_count = 0, 0, 0
    faiss_add_ids, faiss_add_embeddings = [], []
    for email_in_model in emails_input:
        try:
            email_data_dict = email_in_model.model_dump(by_alias=True)
            account_id = email_data_dict.get('account_id')
            if not account_id:
                failed_count += 1
                continue
            existing_email = await get_email_by_id(email_data_dict['id'], account_id=account_id)
            if existing_email and existing_email.get('embedding') and email_data_dict.get('is_unread') == existing_email.get('is_unread'):
                skipped_count += 1
                continue
            if existing_email and existing_email.get('embedding'):
                email_data_dict['skip_embedding'] = True
            processed_metadata = await process_email(email_data_dict)
            if email_data_dict.get('skip_embedding') and existing_email:
                processed_metadata['embedding'] = existing_email.get(
                    'embedding')
            await upsert_email(processed_metadata)
            if not email_data_dict.get('skip_embedding') and 'embedding' in processed_metadata and processed_metadata['embedding']:
                faiss_add_ids.append(processed_metadata['email_id'])
                faiss_add_embeddings.append(processed_metadata['embedding'])
            processed_count += 1
        except Exception as e:
            logger.error(
                f"Error during batch add for email ID {email_in_model.id}: {e}", exc_info=True)
            failed_count += 1
    if faiss_add_ids:
        faiss_index.add(faiss_add_ids, faiss_add_embeddings)
        faiss_index.save()
    return {"processed": processed_count, "failed": failed_count, "skipped": skipped_count, "faiss_total": faiss_index.index.ntotal if faiss_index.index else 0}


@app.patch("/emails/{email_id}/status")
async def mark_email_status(email_id: str, status_update: EmailStatusUpdate, account_id: str):
    if not account_id:
        raise HTTPException(status_code=400, detail="Account ID is required.")
    existing_email = await get_email_by_id(email_id, account_id=account_id)
    if not existing_email:
        raise HTTPException(
            status_code=404, detail="Email not found or access denied.")
    await update_email_status(email_id, status_update.is_unread, account_id)
    return {"message": f"Email {email_id} status updated successfully."}


@app.get("/emails/{email_id}")
async def get_single_email(email_id: str, account_id: str):
    if not account_id:
        raise HTTPException(status_code=400, detail="Account ID is required.")
    email = await get_email_by_id(email_id, account_id=account_id)
    if not email:
        raise HTTPException(
            status_code=404, detail="Email not found or access denied.")
    return email
