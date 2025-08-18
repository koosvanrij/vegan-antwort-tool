import numpy as np
from sentence_transformers import SentenceTransformer
import json

# 1️⃣ Antwortdaten laden (oder direkt aus deiner Python-Liste nehmen)
with open("antworten.json", "r", encoding="utf-8") as f:
    antwort_db = json.load(f)

# 2️⃣ Modell laden
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# 3️⃣ Embeddings berechnen
embeddings = model.encode(antwort_db, convert_to_numpy=True, show_progress_bar=True)

# 4️⃣ Optional: Speicherbedarf halbieren
embeddings = embeddings.astype('float16')

# 5️⃣ Embeddings speichern
np.save("db_embeddings.npy", embeddings)

print("✅ Embeddings gespeichert in db_embeddings.npy")
