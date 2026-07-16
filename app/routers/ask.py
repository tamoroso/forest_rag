from fastapi import APIRouter, status, Depends
from app.schemas import QueryRequest, QueryResponse
from app import get_qdrant_client, get_groq_client
from app.services import generate_rag_response

router = APIRouter(
    prefix="/ask",
    tags=["ask"],
    responses={404: {"description": "Not found"}},
)


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def ask_question(payload: QueryRequest, qdrant_client=Depends(get_qdrant_client), groq_client=Depends(get_groq_client)):
    return await generate_rag_response(payload.question, qdrant_client, groq_client)
