from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

texts = (
    "Hello this is Abhi.",
    "I am learning ge ai",
    "Give me roadmap"
)

vector = embeddings.embed_documents(texts)

print(vector)