import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- 1. CHARGEMENT DE TON ÉCHANTILLON ---

try:
    df = pd.read_csv('ZenAssist_Dataset_1000_Clean.csv')
    print(f"📖 Fichier chargé : {len(df)} lignes.")
except FileNotFoundError:
    print("❌ Erreur : Le fichier 'ZenAssist_Dataset_1000_Clean.csv' est introuvable à la racine.")
    exit(1)

# --- 2. PRÉPARATION DES VARIABLES (Tes colonnes exactes) ---
X = df["Consumer Claim"]
y = df["Tag_clean"]

# Séparation 80/20 avec ton option stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. PREPROCESSING (Tes cellules) ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --- 4. ENTRAÎNEMENT (Le gagnant : Random Forest) ---
model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model_rf.fit(X_train_tfidf, y_train_encoded)

# --- 5. CALCUL DES MÉTRIQUES ---
y_pred = model_rf.predict(X_test_tfidf)
acc = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)

# --- 6. EXPORTATION DES DEUX FICHIERS ---
# A. Le modèle packagé (Model + Vectorizer + Encoder)
model_package = {
    'model': model_rf,
    'tfidf': tfidf,
    'le': le
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

# B. Le rapport JSON pour GitHub Actions
metrics = {
    'accuracy': round(acc * 100, 2),
    'f1_score': round(f1 * 100, 2),
    'model_type': 'Random Forest',
    'samples_used': len(df)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"✅ Terminé ! model.pkl (F1: {metrics['f1_score']}%) et metrics.json générés.")