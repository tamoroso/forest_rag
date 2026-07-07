from fastapi import FastAPI
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform any startup tasks here
    app.state.qdrant = "AsyncQdrantClient(path = 'db')"
    yield
    # Perform any shutdown tasks here

app = FastAPI(lifespan=lifespan)
