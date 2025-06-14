# --- START OF FILE models.py ---

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union
from datetime import datetime

class EmailIn(BaseModel): # Kept for potential other uses, but EmailInput is primary
    id: str
    subject: Optional[str] = None
    from_: Optional[str] = Field(default=None, alias='from')
    body: Optional[str] = None
    date: Optional[str] = None # Should represent received_at
    account_id: str
    is_unread: Optional[bool] = None # Added

    class Config:
        populate_by_name = True

class EmailInput(BaseModel): # This is the one used by /emails/batch_add
    id: str
    threadId: Optional[str] = None # Often provided by email services
    snippet: Optional[str] = None
    subject: Optional[str] = None
    from_: Optional[str] = Field(default=None, alias='from') # 'from' is a reserved keyword
    body: Optional[str] = None # Full body if available
    date: Optional[str] = None # Represents received_at, string initially
    account_id: str
    is_unread: Optional[bool] = None # Added: True if unread, False if read, None if not known

    # For Pydantic v2, model_validator is used. For v1, @root_validator
    @model_validator(mode='before')
    @classmethod
    def ensure_body_or_snippet(cls, values):
        # If 'body' is missing or empty, try to use 'snippet' as 'body'
        # This helps ensure 'body' field in extract_metadata has content if snippet does.
        if not values.get('body') and values.get('snippet'):
            values['body'] = values.get('snippet')
        if not values.get('subject'): # Ensure subject is not None
            values['subject'] = ""
        if not values.get('from_') and values.get('from'):
             values['from_'] = values.get('from')

        return values

    class Config:
        populate_by_name = True
        # For Pydantic V1, use:
        # alias_generator = lambda field_name: 'from' if field_name == 'from_' else field_name
        # allow_population_by_field_name = True

class EmailStatusUpdate(BaseModel):
    is_unread: bool


class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    email_id: Optional[str] = None
    account_id: Optional[str] = None

class ChatMessage(BaseModel):
    conversation_id: str
    role: str
    content: str

# --- END OF FILE models.py ---