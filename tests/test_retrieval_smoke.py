import pytest
from asgi_lifespan import LifespanManager
from app import fastApi_app
from app.services import generate_rag_response


@pytest.mark.asyncio
async def test_generate_rag_response():
    async with LifespanManager(fastApi_app):
        qdrant_client = fastApi_app.state.qdrant_client
        groq_client = fastApi_app.state.groq_client
        question = "Quels sont les leviers d'action pour diminuer le carbone produit par l'agriculture ?"
        result = await generate_rag_response(question, qdrant_client, groq_client)
        print("=== RESULT ===")
        print(result)

        assert result is not None
