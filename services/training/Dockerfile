# 1. Image de base
FROM python:3.12-slim

# 2. Installer uv
RUN pip install --no-cache-dir uv

# 3. Définir le répertoire de travail
WORKDIR /app

# 4. Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# 5. Installer les dépendances lockées
RUN uv sync --locked

# 6. Copier tout le code
COPY . .

# 7. Commande de lancement
CMD ["python", "pipeline/main.py"]
