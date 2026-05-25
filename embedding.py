import os
import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
)
app.prepare(ctx_id=0)

DATASET_DIR = "faces"

def parse_filename(filename):
    stem, _ = os.path.splitext(filename)
    
    # FIX: Handle 'she_her' or 'he_him' containing underscores safely.
    # We explicitly check for the known pronoun strings inside the filename.
    if "_she_her_" in stem:
        gender = "F"
        name = stem.split("_she_her_")[0]
    elif "_he_him_" in stem:
        gender = "M"
        name = stem.split("_he_him_")[0]
    else:
        return None

    return name, gender

# FIX: Use a simple string key (name) to avoid tuple-key tracking vulnerabilities
database = {} 

for root, _dirs, files in os.walk(DATASET_DIR):
    for filename in files:
        metadata = parse_filename(filename)
        if metadata is None:
            print(f"Skipping {filename}: does not match NAME_PRONOUNS_JOB format")
            continue

        name, gender = metadata
        image_path = os.path.join(root, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read {image_path}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face found in {image_path}")
            continue

        # Use the largest detected face in the image.
        face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )

        if name not in database:
            database[name] = {
                "gender": gender,
                "embeddings": []
            }

        database[name]["embeddings"].append(face.embedding)
        print(f"Processed {image_path}")

output = []

for name, data in database.items():
    embeddings = data["embeddings"]
    if len(embeddings) == 0:
        continue

    mean_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm

    output.append({
        "name": name,
        "gender": data["gender"],
        "embedding": mean_embedding.tolist(),
        "sample_count": len(embeddings),
    })

# Save database as JSON
with open("face_db.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Database saved successfully as JSON with normalized embeddings!")