import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. CHARGEMENT
# ==========================================
try:
    df_raw = pd.read_csv('ZenAssist_Dataset_20000_Clean.csv')
    df_raw = df_raw.dropna(subset=['Consumer Claim', 'Tag'])
    print(f"📖 Fichier chargé : {len(df_raw)} lignes.")
except FileNotFoundError:
    print("❌ Erreur : Fichier CSV introuvable.")
    exit(1)

# ==========================================
# 2. MAPPING STRICT (Dataset -> Site Web)
# ==========================================
mapping = {
    # DEBT COLLECTION
    "Debt collection": "Debt collection",
    
    # CONSUMER LOAN
    "Consumer Loan": "Consumer Loan",
    
    # CREDIT CARD
    "Credit card": "Credit card or prepaid card",
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    
    # MORTGAGE
    "Mortgage": "Mortgage",
    
    # VEHICLE
    "Vehicle loan or lease": "Vehicle loan or lease",
    
    # STUDENT
    "Student loan": "Student loan",
    
    # PAYDAY / PERSONAL
    "Payday loan, title loan, or personal loan": "Payday loan, title loan, or personal loan",
    "Payday loan": "Payday loan, title loan, or personal loan",
    
    # CHECKING / SAVINGS
    "Checking or savings account": "Checking or savings account",
    
    # BANK ACCOUNT
    "Bank account or service": "Bank account or service",
    
    # MONEY TRANSFER
    "Money transfer, virtual currency, or money service": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Money transfers": "Money transfers",
    
    # OTHER
    "Other financial service": "Other financial services"
}

# Suppression radicale des réclamations 100% "Credit reporting" du dataset.
# Ainsi, le modèle ne les a plus comme catégorie parasite et devra se concentrer
# sur les autres mots (ex: mortgage, credit card) pour classer ces "claims".
df_raw = df_raw[~df_raw["Tag"].isin([
    "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports"
])]

df_raw['Tag'] = df_raw['Tag'].map(mapping).fillna("Other financial services")

# ==========================================
# 3. RÉEQUILIBRAGE RADICAL (Anti-polarisation)
# ==========================================
# On plafonne chaque classe à 1200 exemples maximum pour qu'aucune ne domine.
max_samples = 1200 
df = df_raw.groupby('Tag').apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42)).reset_index(drop=True)

print("\n📊 Distribution finale envoyée au modèle :")
print(df['Tag'].value_counts())

# ==========================================
# 4. SPLIT & ENCODING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df["Consumer Claim"], df["Tag"], 
    test_size=0.2, random_state=42, stratify=df["Tag"]
)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# ==========================================
# 5. TF-IDF PUISSANT (Trigrams)
# ==========================================
tfidf = TfidfVectorizer(
    max_features=4000, 
    stop_words='english',
    ngram_range=(1, 3), # Capte "credit card" vs "debit card"
    min_df=2,
    dtype=np.float32
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==========================================
# 6. RANDOM FOREST (Sensibilité maximale)
# ==========================================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None, # Arbres complets pour plus de précision
    class_weight="balanced", 
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_tfidf, y_train_encoded)

# ==========================================
# 7. EXPORT
# ==========================================
model_package = {
    'model': model,
    'tfidf': tfidf,
    'le': le
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)

# ==========================================
# 8. ÉVALUATION
# ==========================================
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
size = os.path.getsize('model.pkl') / (1024 * 1024)

metrics = {
    'accuracy': round(acc * 100, 2),
    'f1_score': round(f1 * 100, 2),
    'samples_used': len(df)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"\n🚀 Entraînement terminé !")
print(f"📊 Accuracy: {metrics['accuracy']}% | F1: {metrics['f1_score']}%")
print(f"📦 Taille: {size:.2f} MB")
print("📂 model.pkl et metrics.json mis à jour.")