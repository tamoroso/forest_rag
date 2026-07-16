from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    content: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
