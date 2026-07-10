from fastapi import FastAPI
from contextlib import asynccontextmanager
from app import GROQ_API_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform any startup tasks here
    app.state.qdrant_client = "AsyncQdrantClient(path = 'db')"
    app.state.groq_client = f"AsyncGroq(api_key ={GROK_API_KEY})"
    yield
    # Perform any shutdown tasks here
    await app.state.qdrant_client.close()
    await app.state.groq_client.close()

app = FastAPI(lifespan=lifespan)
