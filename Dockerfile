FROM python:3.11

# Installer Node.js (>=18) et les outils necessaires.
RUN apt-get update \
  && apt-get install -y --no-install-recommends nodejs npm \
  && rm -rf /var/lib/apt/lists/*

# Copier uv depuis l'image officielle.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copier d'abord les manifests de dependances pour profiter du cache Docker.
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Installer les dependances Node et Python.
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync

# Copier le code source du projet.
COPY . .

EXPOSE 3000 5001

# Demarrer simultanement le frontend et le backend en mode developpement.
CMD ["npm", "run", "dev"]
