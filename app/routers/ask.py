from fastapi import APIRouter, status, Depends
from schemas import AskRequest, AskResponse
from app import get_qdrant_client
from services import generate_rag_response

router = APIRouter(
    prefix="/ask",
    tags=["ask"],
    responses={404: {"description": "Not found"}},
)


@router.post("", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_question(payload: AskRequest, qdrant_client=Depends(get_qdrant_client)):
    return await generate_rag_response(payload.question)
