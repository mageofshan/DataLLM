from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.services.router import QueryRouter
from app.services.session_service import SessionService
from app.core.database import get_db

router = APIRouter()
query_router = QueryRouter()

class ChatRequest(BaseModel):
    query: str
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    route: str
    data: Optional[Dict[str, Any]] = None
    session_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Process a natural language query about a dataset.
    """
    try:
        # 1. Get or Create Session
        session_id = request.session_id
        if not session_id:
            session = SessionService.create_session(db, dataset_id=request.dataset_id)
            session_id = session.id
        else:
            # Verify session exists
            session = SessionService.get_session(db, session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

        # 2. Save User Message
        SessionService.add_message(db, session_id, "user", request.query)

        # 3. Route Query & Generate Response
        # TODO: Pass history to router if needed
        result = await query_router.route_query(request.query, request.dataset_id)
        
        # 4. Save Assistant Message
        SessionService.add_message(db, session_id, "assistant", result["response"])

        return ChatResponse(
            response=result["response"],
            route=result["route"],
            data=result.get("data"),
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
