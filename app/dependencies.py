# Requests to recover client instances from global state.
from fastapi import Request


def get_qdrant_client(request: Request):
    return request.app.state.qdrant_client


def get_groq_client(request: Request):
    return request.app.state.groq_client
