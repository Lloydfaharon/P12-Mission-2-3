import pandas as pd
import numpy as np
import pickle
import json

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
# 2. MAPPING (aligné site + corrigé)
# ==========================================
mapping = {
    "Debt collection": "DEBT COLLECTION",
    "Consumer Loan": "CONSUMER LOAN",

    # CREDIT CARD
    "Credit card": "CREDIT CARD OR PREPAID CARD",
    "Credit card or prepaid card": "CREDIT CARD OR PREPAID CARD",
    "Prepaid card": "CREDIT CARD OR PREPAID CARD",

    # MORTGAGE
    "Mortgage": "MORTGAGE",

    # VEHICLE
    "Vehicle loan or lease": "VEHICLE LOAN OR LEASE",

    # STUDENT
    "Student loan": "STUDENT LOAN",

    # PAYDAY
    "Payday loan, title loan, or personal loan": "PAYDAY LOAN, TITLE LOAN, OR PERSONAL LOAN",
    "Payday loan": "PAYDAY LOAN, TITLE LOAN, OR PERSONAL LOAN",

    # BANKING
    "Checking or savings account": "CHECKING OR SAVINGS ACCOUNT",
    "Bank account or service": "BANK ACCOUNT OR SERVICE",

    # MONEY TRANSFER
    "Money transfer, virtual currency, or money service": "MONEY TRANSFER, VIRTUAL CURRENCY, OR MONEY SERVICE",
    "Virtual currency": "MONEY TRANSFER, VIRTUAL CURRENCY, OR MONEY SERVICE",

    # OTHER
    "Money transfers": "MONEY TRANSFERS",
    "Other financial service": "OTHER FINANCIAL SERVICES",

    # FIX IMPORTANT
    "Credit reporting": "OTHER FINANCIAL SERVICES",
    "Credit reporting, credit repair services, or other personal consumer reports": "OTHER FINANCIAL SERVICES"
}

df_raw['Tag'] = df_raw['Tag'].map(mapping).fillna("OTHER FINANCIAL SERVICES")

print("\n📊 Distribution des classes après mapping :")
print(df_raw['Tag'].value_counts())

# ==========================================
# 3. DATASET (on garde toute la data)
# ==========================================
df = df_raw.copy()

# ==========================================
# 4. SPLIT
# ==========================================
X = df["Consumer Claim"]
y = df["Tag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. ENCODING
# ==========================================
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# ==========================================
# 6. TF-IDF (optimisé)
# ==========================================
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),   # 🔥 très important
    min_df=3,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==========================================
# 7. RANDOM FOREST (optimisé)
# ==========================================
model = RandomForestClassifier(
    n_estimators=300,          # + d’arbres = mieux
    max_depth=None,            # laisse apprendre librement
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",       # important pour texte
    class_weight="balanced",   # 🔥 gère le déséquilibre
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_tfidf, y_train_encoded)

# ==========================================
# 8. ÉVALUATION
# ==========================================
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

# ==========================================
# 9. EXPORT MODÈLE
# ==========================================
model_package = {
    'model': model,
    'tfidf': tfidf,
    'le': le
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

# ==========================================
# 10. EXPORT MÉTRIQUES
# ==========================================
metrics = {
    'accuracy': round(acc * 100, 2),
    'f1_score': round(f1 * 100, 2),
    'model_type': 'Random Forest + TF-IDF (12 classes)',
    'samples_used': len(df)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# ==========================================
# 11. RÉSULTATS
# ==========================================
print("\n🚀 Entraînement terminé !")
print(f"📊 Accuracy: {metrics['accuracy']}%")
print(f"🎯 F1-Score: {metrics['f1_score']}%")
print("📂 model.pkl et metrics.json générés avec succès.")