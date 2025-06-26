# --- START OF FILE models.py ---

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

class EmailInput(BaseModel):
    id: str
    threadId: Optional[str] = None
    snippet: Optional[str] = None
    subject: Optional[str] = None
    from_: Optional[str] = Field(default=None, alias='from')
    body: Optional[str] = None
    date: Optional[str] = None
    account_id: str
    is_unread: Optional[bool] = None

    @model_validator(mode='before')
    @classmethod
    def ensure_body_or_snippet(cls, values):
        if not values.get('body') and values.get('snippet'):
            values['body'] = values.get('snippet')
        if not values.get('subject'):
            values['subject'] = ""
        if not values.get('from_') and values.get('from'):
             values['from_'] = values.get('from')
        return values

    class Config:
        populate_by_name = True

class EmailStatusUpdate(BaseModel):
    is_unread: bool

class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    email_id: Optional[str] = None
    # MODIFICATION: account_id is no longer Optional. It is a required string.
    account_id: str

class ChatMessage(BaseModel):
    conversation_id: str
    role: str
    content: str