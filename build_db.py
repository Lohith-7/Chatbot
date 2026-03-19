import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

symptom_df = pd.read_csv('data/Final_Augmented_dataset_Diseases_and_Symptoms.csv')
symptom_df.columns = symptom_df.columns.str.strip().str.lower()
symptom_docs = []

for i in range(len(symptom_df)):
    if i % 100 == 0:
        print(f"Processing row {i}")

    row = symptom_df.iloc[i]
    symptoms = []

    for col in symptom_df.columns:
        if col != "diseases" and row[col] == 1:
            symptoms.append(col.replace("_", " "))

    disease = row["diseases"]

    text = f"Symptoms: {', '.join(symptoms)} → Disease: {disease}"
    symptom_docs.append(text)

unique_docs = {}

for text in symptom_docs:
    disease = text.split("→ Disease:")[-1].strip()
    unique_docs[disease] = text

symptom_docs = list(unique_docs.values())

print(f"\nTotal unique diseases: {len(symptom_docs)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

symptom_db = FAISS.from_texts(symptom_docs, embeddings)
symptom_db.save_local('Symptom_index')

symptom_db.save_local("vector_db/symptom_index")
print("\nSymptom FAISS DB created successfully!")