from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os

def embed_text(text: str, client):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def save_to_pinecone(index, qna_pairs, metadata, client):
    vectors = []
    for pair in qna_pairs:
        q = pair["prompt"].replace("Q: ", "").strip()
        a = pair["completion"].replace("A: ", "").strip()
        text_for_embedding = q + " " + a

        # Use jid + timestamp as unique ID
        timestamp = metadata.get("timestamp")
        vector_id = f"{metadata['group_jid']}_{timestamp}"

        vectors.append({
            "id": vector_id,  # unique id per message timestamp
            "values": embed_text(text_for_embedding, client),
            "metadata": {
                "question": q,
                "answer": a,
                "group": metadata.get("group"),
                "sender": metadata.get("sender"),
                "timestamp": timestamp
            }
        })

    index.upsert(vectors)

def query_pinecone(index, user_query, client, top_k=5):
    query_embedding = embed_text(user_query, client)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches
