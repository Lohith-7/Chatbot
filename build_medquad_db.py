import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load dataset
df = pd.read_csv("data/medquad.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower()

# Remove empty answers
df = df.dropna(subset=["answer"])

docs = []

for i in range(len(df)):
    if i % 1000 == 0:
        print(f"Processing row {i}")

    row = df.iloc[i]

    text = f"""
Disease: {row.get('focus_area', '')}
Question: {row.get('question', '')}
Answer: {row.get('answer', '')}
"""
    docs.append(text)

print(f"\nTotal docs: {len(docs)}")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS DB
print("\nCreating MedQuAD FAISS...")
db = FAISS.from_texts(docs, embeddings)

# Save
db.save_local("vector_db/medquad_index")

print("\nMedQuAD DB created successfully!")