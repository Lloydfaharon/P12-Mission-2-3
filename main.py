from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# 1. CRÉATION DE L'INSTANCE API
app = FastAPI(title="ZenAssist Prediction API")

# 2. CHARGEMENT DU MODÈLE AU DÉMARRAGE
# On charge le dictionnaire qui contient (Modèle + Vectorizer + LabelEncoder)
try:
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    vectorizer = model_data["tfidf"]      
    label_encoder = model_data["le"]
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# 3. DÉFINITION DU FORMAT DES DONNÉES (Pydantic)
class ClaimRequest(BaseModel):
    user_claim: str

# 4. LA ROUTE POST /tags
@app.post("/tags")
async def predict_tags(request: ClaimRequest):
    if not request.user_claim:
        raise HTTPException(status_code=400, detail="La réclamation ne peut pas être vide")
    
    try:
        # Transformation du texte en nombres (TF-IDF)
        X_input = vectorizer.transform([request.user_claim])
        
        # Prédiction (Index numérique)
        prediction_idx = model.predict(X_input)
        
        # Transformation de l'index en nom de Tag (ex: "Credit Card")
        tag_name = label_encoder.inverse_transform(prediction_idx)[0]
        
        return {
            "user_claim": request.user_claim,
            "prediction": tag_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route de santé (Optionnelle mais pro)
@app.get("/")
def home():
    return {"status": "online", "message": "API ZenAssist prête pour les prédictions"}