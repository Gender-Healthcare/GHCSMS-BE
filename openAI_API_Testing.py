# === 8. Use OpenAI Chat (Langchain Agent Equivalent) ===
# def summarize_text(text):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Keep the answer short and suitable for Discord."},
#         {"role": "user", "content": f"{text}"}
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=300
#     )
#     return response['choices'][0]['message']['content']
def check_vector_index(collection_name=MONGO_COLL, index_name=VECTOR_SEARCH):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        # Try a test vector search to verify the index
        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name,
                    'queryVector': [0] * 768,  # Dummy vector of 768 dimensions
                    'path': 'embedding',
                    'numCandidates': 1,
                    'limit': 0
                }
            }
        ]
        list(collection.aggregate(pipeline))  # This will raise an error if the index doesn't exist
        print(f"Vector search index '{index_name}' detected successfully.")
        client.close()
        return True
    except Exception as e:
        print(f"Error checking index: {e}")
        client.close()
        return False