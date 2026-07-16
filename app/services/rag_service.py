from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app import dense_embedding_model_name, collection_name

dense_embedding_model = TextEmbedding(model_name=dense_embedding_model_name)
reranker = TextCrossEncoder(model_name="jinaai/jina-reranker-v2-base-multilingual", threads=1)


async def generate_rag_response(question: str, qdrant_client, groq_client):
    query_embedded = list(dense_embedding_model.query_embed(question))[0]

    initial_retrieval = await qdrant_client.query_points(
        collection_name,
        query=query_embedded,
        with_payload=True,
        limit=5
    )

    hits = []

    for i, hit in enumerate(initial_retrieval.points):
        hit_payload = hit.payload
        hits.append(hit_payload)

    # Reranking
    new_scores = list(
        reranker.rerank(question, [hit["document"] for hit in hits])
    )

    ranking = [
        (i, score) for i, score in enumerate(new_scores)
    ]
    ranking.sort(
        key=lambda x: x[1], reverse=True
    )

    for i, rank in enumerate(ranking):
        print(f'Reranked {i+1} : \"{hits[rank[0]]}\"')

    system_prompt = """Tu es un assistant qui répond uniquement à partir du contexte fourni.
    Règles strictes :
    - Si l'information n'est pas présente dans le contexte, réponds explicitement "Information non trouvée dans les documents fournis."
    - Cite systématiquement la source pour chaque affirmation.
    - Ne mobilise aucune connaissance externe au contexte fourni.
    - Si le rapport distingue plusieurs scénarios, précise toujours à quel scénario un chiffre appartient.
    
    Exemple de réponse : 
    " Il faut décarboner l'agriculture pour limiter le réchauffement climatique [1].
    
    Sources : [1] - Rapport sur l'agriculture
    "

    """

    context_block = "\n".join(
        f'<document source={hits[i]["source"]}>{hits[i]["document"]}</document>' for i, _ in ranking
    )

    user_prompt = f"""Contexte :
    {context_block}

    # Question : {question}"""

    llm_response = await groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

    print(llm_response.choices)

    return {"answer": llm_response.choices[0].message.content, "sources": [{"content": hits[i]["document"], "source": hits[i]["source"]} for i, _ in ranking]}
