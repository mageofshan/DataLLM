from sqlalchemy.orm import Session
from app.models.models import ChatSession, ChatMessage
import uuid
from typing import List, Optional

class SessionService:
    @staticmethod
    def create_session(db: Session, dataset_id: Optional[str] = None, title: str = "New Analysis") -> ChatSession:
        session_id = str(uuid.uuid4())
        db_session = ChatSession(id=session_id, dataset_id=dataset_id, title=title)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session

    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[ChatSession]:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()

    @staticmethod
    def add_message(db: Session, session_id: str, role: str, content: str) -> ChatMessage:
        db_message = ChatMessage(session_id=session_id, role=role, content=content)
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        return db_message

    @staticmethod
    def get_chat_history(db: Session, session_id: str) -> List[ChatMessage]:
        return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
