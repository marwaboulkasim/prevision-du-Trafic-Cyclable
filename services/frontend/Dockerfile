# 1. Image de base
FROM python:3.12-slim

# 2. Installer uv
RUN pip install uv

# 3. Répertoire de travail
WORKDIR /app

# 4. Copier uniquement les fichiers de config en premier (cache optimisé)
COPY pyproject.toml uv.lock ./

# 5. Installer les dépendances
RUN uv sync --frozen

# 6. Copier le reste du code (app.py + assets)
COPY . .

# 7. Exposer le port Streamlit
EXPOSE 8501

# 8. Commande de démarrage
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
