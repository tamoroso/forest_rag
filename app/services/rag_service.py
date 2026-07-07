from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(path="db")

results = qdrant_client.query_points(
    collection_name="shift_project_agriculture",
    query=models.Document(
        text="Comment a évolué la proportion d'élèves issus du milieu agricole dans l'enseignement agricole entre les années 1990 et 2022 ?",
        model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
).points

print(results)
