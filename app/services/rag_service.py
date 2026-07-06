import chromadb

client = chromadb.PersistentClient(path="./db")
collection = client.get_collection("forest_rag")

results = collection.query(
    query_texts=["Comment a évolué la proportion d'élèves issus du milieu agricole dans l'enseignement agricole entre les années 1990 et 2022 ?"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"distance={dist:.3f} | metadata={meta}")
    print(doc[:200])
    print("---")
