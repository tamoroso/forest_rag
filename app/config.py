from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

dense_embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

collection_name = "shift_project_agriculture"
