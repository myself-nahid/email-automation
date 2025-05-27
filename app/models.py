# 1. First, fix your models.py - Update Pydantic configuration

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime

class EmailIn(BaseModel):
    id: str
    subject: str
    from_: str = Field(alias='from')  # Handle 'from' field mapping
    body: str
    date: str
    account_id: str # Added account_id

    class Config:
        populate_by_name = True  # Updated for Pydantic V2

class EmailInput(BaseModel):
    id: str
    threadId: str
    snippet: str
    subject: str
    from_: Optional[str] = Field(default=None, alias='from')
    body: Optional[str] = None
    date: Optional[str] = None
    account_id: str # Added account_id

    class Config:
        populate_by_name = True  # Updated for Pydantic V2

class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    email_id: Optional[str] = None
    account_id: Optional[str] = None # Added to allow querying by account

class ChatMessage(BaseModel):
    conversation_id: str
    role: str
    content: str
