# Usage: poetry run python -m scripts.populate.create_db_snap

from huggingface_hub import login, HfApi

login()

api = HfApi()

api.upload_file(
    path_or_fileobj="snapshot/qdrant_db.tar.gz",
    path_in_repo="qdrant_db.tar.gz",
    repo_id="Amomas/agriculture_shift_project_report_embeddings",
    repo_type="dataset"
)
