FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY scripts/populate/load_db_snap.py ./
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
HF_TOKEN=$(cat /run/secrets/HF_TOKEN) poetry run python load_db_snap.py

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]