import ollama

# Test the embedding dimensions
def test_embedding():
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt='test')
        embedding = response.get('embedding') or response.get('embeddings')
        if embedding:
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"First few values: {embedding[:5]}")
        else:
            print("No embedding returned")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_embedding()