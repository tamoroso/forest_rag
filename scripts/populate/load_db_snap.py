from huggingface_hub import hf_hub_download
import shutil
import os


path = hf_hub_download(repo_id='Amomas/agriculture_shift_project_report_embeddings',
                       filename="qdrant_db.tar.gz", repo_type='dataset', token=os.environ["HF_TOKEN"])

shutil.unpack_archive(path, '.')
