# 1. Utiliser une image Python légère
FROM python:3.10-slim

# 2. Définir le dossier de travail dans le container
WORKDIR /app

# 3. Copier les fichiers nécessaires
COPY requirements.txt .
COPY main.py .
COPY model.pkl .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exposer le port 8000
EXPOSE 8000

# 6. Lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]