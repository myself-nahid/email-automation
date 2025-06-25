# --- START OF FILE main.py ---
from app.openai_client import get_embedding, chat_completion
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from app.models import EmailInput, QuestionRequest, ChatMessage, EmailStatusUpdate
from app.email_processor import process_email, ProcessEmailError
from app.faiss_index import faiss_index
from app.db import (
    database, upsert_email, get_email_by_id, get_emails_by_ids,
    migrate_database, get_emails_in_date_range, update_email_status, metadata, engine
)
from app.chat_manager import add_message, get_chat_history
import faiss
from datetime import datetime, timedelta, timezone
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost", "http://localhost:3000", "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await database.connect()
    metadata.create_all(engine)
    await migrate_database()
    try:
        faiss_index.load()
    except Exception as e:
        logger.error(f"Failed to load FAISS index on startup: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    try:
        faiss_index.save()
    except Exception as e:
        logger.error(f"Failed to save FAISS index on shutdown: {e}", exc_info=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Email Assistant API"}

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def enhanced_truncate_email_content(email: dict, max_chars: int = 1000) -> dict:
    email_copy = email.copy()
    body = str(email_copy.get('body') or '')
    if len(body) <= max_chars:
        email_copy['body'] = body
        return email_copy
    priority_keywords = [
        'meeting', 'interview', 'call', 'appointment', 'zoom', 'teams', 'schedule', 'invite',
        'time', 'date', 'urgent', 'important', 'action required', 'deadline', 'asap',
        'invoice', 'payment', 'receipt', 'confirm', 'booking'
    ]
    sentences = re.split(r'(?<=[.!?])\s+', body)
    prioritized_sentences, other_sentences = [], []
    for sentence in sentences:
        if not sentence.strip(): continue
        sentence_lower = sentence.lower()
        has_date = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|\b\d{1,2}[/-]\d{1,2}\b', sentence_lower)
        has_time = re.search(r'\b\d{1,2}:\d{2}\s*(am|pm)?\b', sentence_lower)
        if any(kw in sentence_lower for kw in priority_keywords) or has_date or has_time:
            prioritized_sentences.append(sentence)
        else:
            other_sentences.append(sentence)
    truncated_parts = []
    current_len = 0
    ellipsis = "..."
    for s_list in [prioritized_sentences, other_sentences]:
        for s in s_list:
            if current_len + len(s) < max_chars - len(ellipsis):
                truncated_parts.append(s)
                current_len += len(s) + 1
            else: break
    if not truncated_parts:
        truncated_body = body[:max_chars - len(ellipsis)] + ellipsis
    else:
        truncated_body = ' '.join(truncated_parts) + ellipsis
    email_copy['body'] = truncated_body
    return email_copy

def enhanced_build_smart_context(emails: List[dict], max_total_tokens=4000) -> str:
    if not emails:
        return "No relevant emails found for the query."
    context_parts = []
    total_tokens = 0
    for i, email in enumerate(emails, 1):
        date_str = email.get('received_at', 'Unknown date')
        if isinstance(date_str, datetime): date_str = date_str.strftime('%Y-%m-%d %H:%M UTC')
        subject = str(email.get('subject', 'No subject')).strip()
        sender = str(email.get('from_', 'Unknown sender')).strip()
        body = str(email.get('body', '')).strip()
        category = str(email.get('category', 'general')).capitalize()
        status = "Unknown"
        if email.get('is_unread') is True: status = "Unread"
        elif email.get('is_unread') is False: status = "Read"
        part = f"""--- EMAIL {i} ---
Email ID: {email.get('email_id')}
Date: {date_str}
From: {sender}
Subject: {subject}
Status: {status}
Category: {category}
Content Snippet: {body}
"""
        estimated_tokens_part = estimate_tokens(part)
        if total_tokens + estimated_tokens_part > max_total_tokens: break
        context_parts.append(part)
        total_tokens += estimated_tokens_part
    return "\n".join(context_parts)

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
            
            needs_embedding = True
            if existing_email and existing_email.get('embedding'):
                needs_embedding = False
            
            # If only unread status changed, we can avoid re-processing for embedding
            if existing_email and not needs_embedding and email_data_dict.get('is_unread') != existing_email.get('is_unread'):
                 email_data_dict['skip_embedding'] = True

            # If email is new or needs an embedding, process fully.
            # Otherwise, we might just be updating the unread status.
            processed_metadata = await process_email(email_data_dict)
            
            if email_data_dict.get('skip_embedding') and existing_email:
                processed_metadata['embedding'] = existing_email.get('embedding')
            
            await upsert_email(processed_metadata)
            
            if not email_data_dict.get('skip_embedding') and 'embedding' in processed_metadata and processed_metadata['embedding']:
                faiss_add_ids.append(processed_metadata['email_id'])
                faiss_add_embeddings.append(processed_metadata['embedding'])
            
            processed_count += 1
        except Exception as e:
            logger.error(f"Error during batch add for email ID {email_in_model.id}: {e}", exc_info=True)
            failed_count += 1
            
    if faiss_add_ids:
        faiss_index.add(faiss_add_ids, faiss_add_embeddings)
        faiss_index.save()
        logger.info(f"Added {len(faiss_add_ids)} new embeddings to FAISS and saved.")

    return {
        "processed": processed_count, "failed": failed_count, "skipped": skipped_count,
        "faiss_total": faiss_index.index.ntotal if faiss_index.index else 0
    }

CONTEXT_TOKEN_LIMIT = 4000
MAX_EMAILS_TO_FETCH = 50

@app.post("/query/working")
async def working_query(request: QuestionRequest):
    try:
        emails_for_context = []
        account_id = request.account_id
        question_lower = request.question.lower()
        logger.info(f"New query for account '{account_id}': '{request.question}'")

        retrieved_ids = set()
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=90)
        
        category_to_check = None
        if any(k in question_lower for k in ['urgent', 'important', 'asap', 'deadline']):
            category_to_check = 'urgent'
        elif any(k in question_lower for k in ['meeting', 'event', 'call', 'appointment', 'schedule']):
            category_to_check = 'meeting/event'
        elif any(k in question_lower for k in ['invoice', 'payment', 'receipt']):
            category_to_check = 'financial'

        # Strategy 1: High-precision category search
        if category_to_check:
            logger.info(f"Strategy 1: Querying for category '{category_to_check}'")
            db_emails = await get_emails_in_date_range(
                start_date, end_date, account_id=account_id, category=category_to_check, limit=MAX_EMAILS_TO_FETCH
            )
            for row in db_emails:
                if row['email_id'] not in retrieved_ids:
                    emails_for_context.append(dict(row))
                    retrieved_ids.add(row['email_id'])
            logger.info(f"Found {len(db_emails)} emails in category '{category_to_check}'.")

        # Strategy 2: Semantic search for nuance
        logger.info("Strategy 2: Performing semantic search with FAISS.")
        try:
            q_embedding_data = await process_email({'id': 'query', 'account_id': account_id, 'subject': request.question, 'body': request.question}, generate_embedding_only=True)
            if q_embedding_data and q_embedding_data.get('embedding'):
                embedding_array = np.frombuffer(q_embedding_data['embedding'], dtype=np.float32)
                distances, top_faiss_ids = faiss_index.search_with_scores(embedding_array, top_k=MAX_EMAILS_TO_FETCH)
                
                logger.info(f"FAISS returned {len(top_faiss_ids)} IDs.")
                if top_faiss_ids:
                    faiss_results = await get_emails_by_ids(top_faiss_ids, account_id=account_id)
                    for row in faiss_results:
                         if row['email_id'] not in retrieved_ids:
                            emails_for_context.append(dict(row))
                            retrieved_ids.add(row['email_id'])
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)

        # Strategy 3: Fallback to general recent emails
        if not emails_for_context:
            logger.info("Strategy 3: No results yet, falling back to general recent emails.")
            db_emails = await get_emails_in_date_range(
                start_date, end_date, account_id=account_id, limit=20
            )
            for row in db_emails:
                if row['email_id'] not in retrieved_ids:
                    emails_for_context.append(dict(row))
                    retrieved_ids.add(row['email_id'])
            logger.info(f"Found {len(db_emails)} recent emails as fallback.")

        if not emails_for_context:
            return {"answer": "I couldn't find any relevant emails to answer your question. Your inbox might be empty or the emails might be older than 90 days."}
        
        emails_for_context.sort(key=lambda x: x['received_at'], reverse=True)

        logger.info(f"Retrieved a total of {len(emails_for_context)} unique candidate emails. Preparing context for AI.")
        truncated_emails = [enhanced_truncate_email_content(e.copy()) for e in emails_for_context]
        context_text = enhanced_build_smart_context(truncated_emails, max_total_tokens=CONTEXT_TOKEN_LIMIT)

        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        system_prompt = f"""You are a highly intelligent AI Email Assistant. Your goal is to answer the user's questions based *only* on the email data provided below.
Current date and time is: {current_time}. Use this to determine if an event is in the past, present, or future.

**ANALYSIS INSTRUCTIONS:**
1.  **Strictly Adhere to Data:** Base your answer *exclusively* on the content of the emails provided. Do not invent information. If the answer isn't in the emails, state that clearly.
2.  **Prioritize by Category:** Pay close attention to the `Category` of each email. For questions about meetings, prioritize `Category: Meeting/event`. For urgent matters, prioritize `Category: Urgent`.
3.  **Use Email Status:** Use the `Status` (Unread/Read) field to answer questions about which emails the user has or hasn't seen.
4.  **Be Specific and Actionable:** Extract key details like dates, times, locations, required actions, and deadlines.
5.  **Format for Clarity:** Use lists, bullet points, and bold text to make your answers easy to read.

**EMAIL DATA FOR ANALYSIS:**
{context_text}

Now, answer the user's question.
"""
        messages = []
        if request.conversation_id:
            history = await get_chat_history(request.conversation_id)
            history_messages = [{"role": msg['role'], "content": msg['content']} for msg in history[-4:]]
            messages.extend(history_messages)

        messages.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ])
        
        answer = await chat_completion(messages)
        if not answer:
            answer = "I apologize, but I was unable to generate a response. Please try rephrasing your question."

        if request.conversation_id:
            await add_message(request.conversation_id, "user", request.question)
            await add_message(request.conversation_id, "assistant", answer)

        return {"answer": answer}

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")

@app.patch("/emails/{email_id}/status")
async def mark_email_status(email_id: str, status_update: EmailStatusUpdate, account_id: str):
    existing_email = await get_email_by_id(email_id, account_id=account_id)
    if not existing_email:
        raise HTTPException(status_code=404, detail="Email not found or access denied.")
    await update_email_status(email_id, status_update.is_unread, account_id)
    return {"message": f"Email {email_id} status updated successfully."}

@app.get("/emails/{email_id}")
async def get_single_email(email_id: str, account_id: str):
    email = await get_email_by_id(email_id, account_id=account_id)
    if not email:
        raise HTTPException(status_code=404, detail="Email not found or access denied.")
    return email